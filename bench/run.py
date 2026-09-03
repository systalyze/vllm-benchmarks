#!/usr/bin/env python3
"""Reproduce ONE benchmark deployment from a clean clone.

    bench/run.sh <config folder> [--arm dspark|eagle3] [--gpus 0,1] [--out DIR] [--dry-run]

bench/run.sh resolves the config folder to its row.json + launch.sh (and checks
out the vLLM branch the launch.sh names in the vllm/ submodule) before calling
this file, which is the campaign package's run.py with only its paths changed:
a config's files are given explicitly, and any "$REPO/..." path in row.json
(patch mounts from the vllm/ submodule, the draft under models/) is resolved
against this repository's root.

For the given row.json this
  (a) prints the exact docker run line(s) the row boots,
  (b) boots them, (c) waits for /health on every replica,
  (d) runs the lambda_sat probe (closed loop, c=64, 180 s, one replica), the
      open-loop Poisson ladder at 0.3/0.5/0.7/0.85/1.0 x lambda_sat and then the
      knee ladder (+0.1 x lambda_sat per rung to the confirmed knee) via
      bench/run_replicas.py, with the row's own workload flags
      (curve rows: workload/aws-p50p90-v1 with ignore_eos:true temperature:0;
      speculator rows and their off-controls: the exact-length variant
      workload/aws-p50p90-v1-mintok, per-request min_tokens = max_tokens, no
      ignore_eos, temperature:0 - the protocol of record, METHODOLOGY §6;
      --workload-dir overrides either default),
  (e) writes <out>/<row_id>/cells/rate_*.json, sweep_aggregate/ and
      aws_sweep_manifest.json (plus per-replica AIPerf artifacts under
      <out>/<row_id>-replicas/), and
  (f) stops and removes the containers, by name, whether or not it succeeded.

Steps (b)-(f) are bench/run_replicas.py, unchanged from the campaign; this
wrapper only resolves the deployment, applies --gpus, resolves the protocol
knobs (--until-knee, --requests-per-rate, --warmup-requests) and prints the plan.
--dry-run prints every command that would run and executes none of them.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
sys.path.insert(0, str(BENCH))

import run_replicas as rr  # noqa: E402  (bench/run_replicas.py)

DEFAULT_WORKLOAD = REPO / "workload" / "aws-p50p90-v1"
# Speculator rows and their off-controls hold every request to its sampled OSL with
# per-request min_tokens = max_tokens instead of ignore_eos (METHODOLOGY §6); the
# per-request value lives in this workload variant's files, not in a flag.
MINTOK_WORKLOAD = REPO / "workload" / "aws-p50p90-v1-mintok"

# A rung's in-flight count can never exceed its request count, so at or above
# this max-num-seqs the campaign's 160-request rung measured the rung size and
# not the server; those deployments are measured at 512 requests + 16 warm-ups
# (their row.json records requests_per_rate 512 / warmup_requests 16).
LARGE_BATCH_MAX_NUM_SEQS = 128
LARGE_BATCH_REQUESTS_PER_RATE = 512
LARGE_BATCH_WARMUP_REQUESTS = 16
VLLM_DEFAULT_MAX_NUM_SEQS = 256  # what a row that passes no --max-num-seqs runs at


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("row_file", type=Path, help="the config's row.json (bench/run.sh passes it)")
    p.add_argument("--launch", type=Path, default=None,
                   help="the launch.sh beside it, whose docker run line(s) are echoed for the record")
    p.add_argument("--gpus", help="comma-separated GPU indices to use instead of the row's own list; "
                                  "must be a multiple of the row's TP (one replica per TP-group)")
    p.add_argument("--out", type=Path, default=None, help="output directory (default: runs/<row_id>-<UTC stamp>)")
    p.add_argument("--workload-dir", type=Path, default=None,
                   help="frozen workload directory (default: workload/aws-p50p90-v1-mintok, the "
                        "exact-length variant, for a row that carries a speculative_config or that "
                        "drops ignore_eos as a speculator off-control; workload/aws-p50p90-v1 "
                        "otherwise); an explicit directory always wins")
    p.add_argument("--base-port", type=int, default=8300)
    p.add_argument("--health-timeout", type=int, default=1800)
    p.add_argument("--probe-concurrency", type=int, default=None,
                   help="override the c=64 lambda_sat probe (the campaign used 128 on some stock rows)")
    p.add_argument("--rate-ratios", default=None,
                   help="override the fixed rungs (default: the row's own rate_ratios if recorded, "
                        "else 0.3,0.5,0.7,0.85,1.0)")
    p.add_argument("--until-knee", action=argparse.BooleanOptionalAction, default=True,
                   help="after the fixed rungs, step +0.1 x lambda_sat to the confirmed plateau (the "
                        "on by default: every capacity is a knee-ladder "
                        "number); --no-until-knee runs only the fixed rungs")
    p.add_argument("--requests-per-rate", type=int, default=None,
                   help="requests per rung per replica (default: the row's own requests_per_rate if "
                        f"recorded, else {LARGE_BATCH_REQUESTS_PER_RATE} when its max-num-seqs is "
                        f">= {LARGE_BATCH_MAX_NUM_SEQS} and {rr.REQUESTS_PER_CELL} otherwise)")
    p.add_argument("--warmup-requests", type=int, default=None,
                   help="warm-ups per rung per replica (default: the row's own warmup_requests if "
                        f"recorded, else {LARGE_BATCH_WARMUP_REQUESTS} for a large-batch row and "
                        f"{rr.WARMUP_REQUESTS} otherwise)")
    p.add_argument("--python", default=None,
                   help="interpreter that has aiperf 0.12.0 installed (default: bench/.venv/bin/python if present, else this one)")
    p.add_argument("--dry-run", action="store_true", help="print every command; execute nothing")
    return p.parse_args()


def load_row(path: Path) -> dict:
    if not path.is_file():
        sys.exit(f"no such file: {path}\n(see configs/INDEX.md)")
    return json.loads(path.read_text(encoding="utf-8"))


def expand_repo(value):
    """Replace "$REPO" with this repository's root in every string of a row."""
    if isinstance(value, str):
        return value.replace("$REPO", str(REPO))
    if isinstance(value, list):
        return [expand_repo(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_repo(v) for k, v in value.items()}
    return value


def effective_row(row: dict, gpus_arg: str | None) -> dict:
    listed_gpus = list(row["gpus"])
    row = expand_repo(dict(row))
    tp = int(row["tp"])
    if gpus_arg:
        gpus = [int(g) for g in gpus_arg.split(",") if g.strip()]
        if not gpus or len(gpus) % tp:
            sys.exit(f"--gpus must list a multiple of TP={tp} GPUs; got {gpus}")
        row["gpus"] = gpus
        row["gpus_override"] = f"--gpus {gpus_arg} (row.json listed {listed_gpus})"
    fm = row.get("fork_mounts")
    if fm and not fm["source_root"].startswith("/"):
        fm = dict(fm)
        fm["source_root"] = str(REPO / fm["source_root"])
        row["fork_mounts"] = fm
    return row


def choose_python(explicit: str | None) -> str:
    if explicit:
        return explicit
    venv = BENCH / ".venv" / "bin" / "python"
    return str(venv) if venv.is_file() else sys.executable


def is_speculator_row(row: dict) -> bool:
    """A row that serves a draft: `speculative_config` in row.json, or --speculative-config on its argv."""
    return bool(row.get("speculative_config")) or "--speculative-config" in (row.get("vllm_args") or [])


def is_speculator_off_control(row: dict) -> bool:
    """The no-draft control of a speculator arm: same deployment, and the row drops ignore_eos
    so both arms see the identical exact-length token count (METHODOLOGY §6)."""
    overrides = row.get("workload_overrides") or {}
    return overrides.get("ignore_eos") is False and not is_speculator_row(row)


def workload_choice(row: dict, explicit: Path | None) -> tuple[Path, str]:
    """The workload directory this reproduction drives, and the one-line reason for it."""
    if explicit is not None:
        return explicit, f"--workload-dir {explicit} given on the command line (overrides the row's default)"
    if is_speculator_row(row):
        return MINTOK_WORKLOAD, ("speculator row: exact-length workload aws-p50p90-v1-mintok, no ignore_eos "
                                 "(per-request min_tokens = max_tokens = sampled OSL; METHODOLOGY §6)")
    if is_speculator_off_control(row):
        return MINTOK_WORKLOAD, ("speculator off-control: exact-length workload aws-p50p90-v1-mintok, no "
                                 "ignore_eos (same file as its draft arm, so both see the identical token count)")
    return DEFAULT_WORKLOAD, "curve row: workload aws-p50p90-v1, ignore_eos:true (forced OSL), temperature:0"


def row_max_num_seqs(row: dict) -> int:
    """The deployment's own batch cap: its --max-num-seqs, or vLLM's default when it passes none."""
    vllm_args = row["vllm_args"]
    if "--max-num-seqs" in vllm_args:
        return int(vllm_args[vllm_args.index("--max-num-seqs") + 1])
    return VLLM_DEFAULT_MAX_NUM_SEQS


def protocol(row: dict, args: argparse.Namespace) -> dict:
    """The ladder knobs this reproduction runs with: an explicit flag wins, then what the
    row itself recorded when it was measured, then the current protocol's default for a
    deployment of its batch size."""
    ratios = args.rate_ratios or (",".join(str(r) for r in row["rate_ratios"]) if row.get("rate_ratios") else None)
    large_batch = row_max_num_seqs(row) >= LARGE_BATCH_MAX_NUM_SEQS
    return {
        "ratios": ratios,
        "ratio_list": [float(x) for x in ratios.split(",")] if ratios else list(rr.RATE_RATIOS),
        "requests_per_rate": (args.requests_per_rate or row.get("requests_per_rate")
                              or (LARGE_BATCH_REQUESTS_PER_RATE if large_batch else rr.REQUESTS_PER_CELL)),
        "warmup_requests": (args.warmup_requests or row.get("warmup_requests")
                            or (LARGE_BATCH_WARMUP_REQUESTS if large_batch else rr.WARMUP_REQUESTS)),
        "probe_concurrency": args.probe_concurrency or rr.PROBE_CONCURRENCY,
        "until_knee": args.until_knee,
        "max_num_seqs": row_max_num_seqs(row),
    }


def plan(row: dict, args: argparse.Namespace, out: Path, python: str, row_file: Path,
         proto: dict) -> list[str]:
    """Every command run_replicas.py will issue for this row, in order."""
    lines: list[str] = []
    groups = rr.replica_groups(list(row["gpus"]), int(row["tp"]))
    ports = [args.base_port + i for i in range(len(groups))]
    workload_file = args.workload_dir / f"requests-{row['tokenizer_id']}.jsonl"
    extra_inputs = rr.workload_extra_inputs(row)
    ratio_list = proto["ratio_list"]
    probe_c = proto["probe_concurrency"]

    lines.append(f"# row {row['row_id']}: {len(groups)} replica(s) x TP{row['tp']} on GPUs {row['gpus']}, "
                 f"image {row['image']}, provisioned_gpus={row.get('provisioned_gpus', len(row['gpus']))}, "
                 f"full_node_gpus={row.get('full_node_gpus', len(row['gpus']))}")
    lines.append(f"# (d0) {args.workload_reason}")
    lines.append(f"# workload {workload_file}  extra-inputs {extra_inputs}")
    lines.append("# preflight: nvidia-smi must show < 512 MiB used on every listed GPU; ports must be free; "
                 "container names must not exist")
    lines.append("")
    lines.append("# (a)+(b) boot")
    for group, port in zip(groups, ports):
        name = rr.container_name(row["row_id"], group)
        cmd = ["docker", "run", "-d", "--name", name,
               "--gpus", f'"device={",".join(str(g) for g in group)}"',
               "--ipc=host", "--shm-size=16g", "-p", f"{port}:{port}", "-v", "/data:/data"]
        cmd += row.get("docker_run_args") or []
        for k, v in (row.get("env") or {}).items():
            cmd += ["-e", f"{k}={v}"]
        fm = row.get("fork_mounts")
        if fm:
            root = fm["target_root"].rstrip("/")
            for rel in fm["files"]:
                cmd += ["-v", f"{fm['source_root'].rstrip('/')}/{rel}:{root}/{rel.removeprefix('vllm/')}:ro"]
        cmd += [row["image"], "--model", row["model_path"], "--served-model-name", row["served_model_name"],
                "--tensor-parallel-size", str(row["tp"]), "--port", str(port), "--host", "0.0.0.0"]
        cmd += row.get("vllm_args") or []
        lines.append(shlex.join(cmd))
    lines.append("")
    lines.append("# (c) health: poll every 5 s until HTTP 200, abort if the container exits or "
                 f"{args.health_timeout} s pass")
    for group, port in zip(groups, ports):
        lines.append(f"curl -sf http://127.0.0.1:{port}/health")
    if row.get("fork_mounts"):
        lines.append("# fork/patch rows: sha256sum of every mounted file INSIDE the container is compared with the host copy")
    lines.append("")
    if row["lambda_sat"] == "probe":
        lines.append(f"# (d1) lambda_sat probe on replica 1: closed loop c={probe_c}, {rr.PROBE_SECONDS} s, "
                     f"{proto['warmup_requests']} warmups at c=8; lambda_sat = output tok/s / mean OSL "
                     f"(the campaign probed a large-batch row at c = max(64, 2 x max-num-seqs); this row's "
                     f"max-num-seqs is {proto['max_num_seqs']}, pass --probe-concurrency to match it)")
        lines.append(shlex.join(rr.aiperf_command(config=out / row["row_id"] / "probe" / "aiperf_probe.yaml",
                                                  tokenizer=row["tokenizer_path"],
                                                  base_url=f"http://127.0.0.1:{ports[0]}",
                                                  extra_inputs=extra_inputs)))
        lam = "<lambda_sat from probe>"
    else:
        if not isinstance(row["lambda_sat"], (int, float)):
            sys.exit(f"{row['row_id']}: row.json states lambda_sat {row['lambda_sat']!r}, not a rate "
                     f"and not 'probe' — this row's ladder was pinned to another row that had not been "
                     f"probed when it was written ({row.get('lambda_sat_source')}). Run again with "
                     f"--rate-ratios against a rate you supply, or re-export the row.")
        lam = f"{float(row['lambda_sat']):.6g}"
        lines.append(f"# (d1) lambda_sat PINNED by row.json to {lam} req/s per replica (no probe)")
    lines.append("")
    lines.append(f"# (d2) fixed rungs: ratios {ratio_list} x lambda_sat, {proto['requests_per_rate']} requests + "
                 f"{proto['warmup_requests']} warmups per replica per rung, Poisson arrivals, no client concurrency "
                 "cap; all replicas run each rung concurrently")
    for ratio in ratio_list:
        rate = f"{float(lam) * ratio:.6g}" if lam[0] != "<" else f"{ratio}*{lam}"
        for group, port in zip(groups, ports):
            cmd = [python, str(BENCH / "run_rate_sweep.py"),
                   "--config-id", f"gpu{group[0]}", "--model", row["served_model_name"],
                   "--tokenizer", row["tokenizer_path"],
                   "--url", f"http://127.0.0.1:{port}/v1/chat/completions",
                   "--gpu-count", str(len(group)), "--rates", rate,
                   "--workload-file", str(workload_file), "--workload-id", args.workload_dir.name,
                   "--requests-per-rate", str(proto["requests_per_rate"]),
                   "--warmup-requests", str(proto["warmup_requests"]),
                   "--runs", "1", "--ui", "none",
                   "--artifact-root", str(out / f"{row['row_id']}-replicas" / f"rate_{rate}"),
                   "--server-container", rr.container_name(row["row_id"], group),
                   "--no-default-extra-inputs"]
            for e in extra_inputs:
                cmd += ["--extra-input", e]
            lines.append(shlex.join(cmd))
    lines.append("")
    if proto["until_knee"]:
        lines.append(f"# (d3) knee ladder: +{rr.LADDER_STEP} x lambda_sat per rung from "
                     f"{rr.LADDER_START_RATIO} x, same requests and warmups, stopping at the confirmed "
                     f"knee (first rung whose tail-corrected achieved rate is below "
                     f"{rr.KNEE_ACHIEVED_FRACTION:.0%} of the offered rate, then one confirmation rung at "
                     f"{rr.CONFIRMATION_FACTOR} x it), at a plateau, or on any errored request; "
                     "one run_rate_sweep.py call per rung per replica, exactly as above")
    else:
        lines.append("# (d3) knee ladder: DISABLED by --no-until-knee (the fixed rungs only; the "
                     "capacities are knee-ladder numbers, so this reproduces a partial ladder)")
    lines.append("")
    lines.append(f"# (e) written by run_replicas.py: {out}/{row['row_id']}/cells/rate_*.json, "
                 f"sweep_aggregate/profile_export_aiperf_sweep.json, aws_sweep_manifest.json, "
                 f"launch-*.txt, docker-logs/; DONE or ABORTED marker in {out}")
    lines.append("")
    lines.append("# (f) teardown, by exact name, in a finally: block")
    for group in groups:
        lines.append(f"docker stop {rr.container_name(row['row_id'], group)}")
    lines.append("")
    lines.append("# the single command this wrapper actually executes for (b)-(f):")
    lines.append(shlex.join(driver_command(row_file, args, out, python, proto)))
    return lines


def driver_command(row_file: Path, args: argparse.Namespace, out: Path, python: str,
                   proto: dict) -> list[str]:
    cmd = [python, str(BENCH / "run_replicas.py"),
           "--row", str(row_file), "--out-dir", str(out),
           "--workload-dir", str(args.workload_dir),
           "--driver", str(BENCH / "run_rate_sweep.py"),
           "--base-port", str(args.base_port),
           "--health-timeout", str(args.health_timeout),
           "--requests-per-rate", str(proto["requests_per_rate"]),
           "--warmup-requests", str(proto["warmup_requests"])]
    if args.probe_concurrency:
        cmd += ["--probe-concurrency", str(args.probe_concurrency)]
    if proto["ratios"]:
        cmd += ["--rate-ratios", proto["ratios"]]
    if proto["until_knee"]:
        cmd += ["--until-knee"]
    return cmd


def preflight(row: dict, args: argparse.Namespace, python: str) -> None:
    problems = []
    for tool in ("docker", "nvidia-smi"):
        if not shutil.which(tool):
            problems.append(f"{tool} not on PATH")
    if not Path(row["model_path"]).is_dir():
        problems.append(f"checkpoint missing: {row['model_path']} (see checkpoints.md)")
    sc = row.get("speculative_config") or {}
    if sc.get("model") and not Path(sc["model"]).is_dir():
        problems.append(f"draft checkpoint missing: {sc['model']} (see checkpoints.md)")
    wf = args.workload_dir / f"requests-{row['tokenizer_id']}.jsonl"
    if not wf.is_file():
        problems.append(f"workload file missing: {wf}")
    fm = row.get("fork_mounts")
    if fm:
        for rel in fm["files"]:
            if not (Path(fm["source_root"]) / rel).is_file():
                problems.append(f"patch file missing: {Path(fm['source_root']) / rel}")
    for k, v in (row.get("env") or {}).items():
        if k == "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR" and not Path(v).is_dir():
            problems.append(f"{k}={v} is a host directory under /data that must exist (mkdir -p {v}); "
                            "it is a warm-start cache, empty is fine")
    check = subprocess.run([python, "-c", "import aiperf, yaml; print(aiperf.__version__)"],
                           capture_output=True, text=True)
    if check.returncode != 0 or check.stdout.strip() != "0.12.0":
        problems.append(f"{python} lacks aiperf==0.12.0 (+pyyaml): pip install -r bench/requirements.txt "
                        f"(got: {check.stdout.strip() or check.stderr.strip()[-200:]})")
    if problems:
        sys.exit("preflight failed:\n  - " + "\n  - ".join(problems))


def main() -> int:
    args = parse_args()
    row = effective_row(load_row(args.row_file), args.gpus)
    python = choose_python(args.python)
    import datetime as dt
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out = (args.out or REPO / "runs" / f"{row['row_id']}-{stamp}").resolve()
    args.workload_dir, args.workload_reason = workload_choice(row, args.workload_dir)
    args.workload_dir = args.workload_dir.resolve()
    row_file = out / f"{row['row_id']}.row.json"
    proto = protocol(row, args)

    launch = args.launch or args.row_file.parent / "launch.sh"
    if launch.is_file():
        print(f"# launch line(s) as committed in {launch} (BASE_PORT=8300):")
        for line in launch.read_text(encoding="utf-8").splitlines():
            if line.startswith("docker run"):
                print(line)
        print()
    print("\n".join(plan(row, args, out, python, row_file, proto)))
    if args.dry_run:
        print("\n# --dry-run: nothing executed")
        return 0

    preflight(row, args, python)
    out.mkdir(parents=True, exist_ok=True)
    row_file.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    cmd = driver_command(row_file, args, out, python, proto)
    print(f"\n# executing: {shlex.join(cmd)}\n", flush=True)
    env = dict(os.environ)
    env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", "")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
