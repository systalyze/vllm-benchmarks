#!/usr/bin/env python3
"""Run one open-loop rate sweep per fixed deployment row, replica-parallel.

One row = one fixed vLLM deployment (model, image, argv, TP, GPU set). TP=1
means one container per GPU; TP=2 means one per GPU pair. Every replica is
benchmarked concurrently by its own AIPerf process against the frozen workload
list, and the cell is reported at node scope: node output tok/s is the sum over
replicas, per-GPU divides by the PROVISIONED GPU count, and TTFT/ITL
percentiles are pooled over every replica's per-request records.

Containers are named aws-<row>-gpu<first gpu> and are stopped only by that
name. A GPU that already holds memory is refused, never reused.

Every benchmark cell asserts its AIPerf artifact exists. A missing artifact,
a boot failure or a health timeout aborts the whole run and writes an ABORTED
marker. A clean finish over every row writes DONE, which the outer runner
watches to stop the instance.

Knee-finding flags (2026-09-02, "Saturation and re-run plan", section a;
the ladder itself is run_rate_sweep.knee_ladder, one run_rate_sweep.py call
per rung per replica exactly as the fixed rungs are run today):

  --until-knee          after the fixed ratios, step +0.1 x lambda_sat from
                        --start-ratio (1.1) and stop at the confirmed knee:
                        knee = first rung whose achieved request rate is below
                        95 % of the offered rate; one confirmation rung at
                        1.5 x the knee rate; output tok/s within 3 % of the
                        knee's -> stop "plateau", higher -> knee cleared and
                        +0.1 stepping resumes from the confirmation ratio;
                        two consecutive +0.1 rungs gaining < 3 % tok/s ->
                        stop "plateau-before-knee"; any errored request ->
                        stop "errors". knee_rate, knee_ratio, stop_reason,
                        rungs_run go into aws_sweep_manifest.json; every rung
                        is a cells/rate_<x>.json like the fixed ones.
  --knee-achieved       tail-corrected (default) or raw; see run_rate_sweep.py.
  --no-probe            never run the lambda_sat probe; the row must carry a
                        number, or --lambda-sat must be given.
  --lambda-sat <req/s>  use this saturation rate (e.g. the row's earlier
                        manifest, or 2 x it for a high-budget re-run).
  --extend-from <dir>   a finished row directory (aws_sweep_manifest.json +
                        cells/): boot the same deployment and run ONLY the
                        ladder, no fixed ratios, writing new cells/rate_<x>.json
                        beside the old ones (an existing rate file is never
                        overwritten) and updating the manifest in place (the
                        previous manifest is kept as
                        aws_sweep_manifest.pre-extension-<stamp>.json).
                        lambda_sat comes from that manifest unless --lambda-sat
                        is given; boot receipts go under extension-<stamp>/;
                        the markers are EXTENSION-DONE-/EXTENSION-ABORTED-<stamp>
                        in that directory.
  --skip-ratio 0.3      drop a fixed ratio (repeatable); default runs all five.
  --gpus 0,1            run on these GPUs instead of the row's (extension of a
                        row whose GPUs are now busy).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_rate_sweep import (  # the ladder logic lives in the rate driver
    CONFIRMATION_FACTOR,
    KNEE_ACHIEVED_BASES,
    KNEE_ACHIEVED_FRACTION,
    LADDER_START_RATIO,
    LADDER_STEP,
    MAX_LADDER_RUNGS,
    PLATEAU_TOLERANCE,
    knee_ladder,
    rung_timing,
    tail_corrected_qps,
)
from workload_variant import check_extra_inputs  # scripts/workload_variant.py

RATE_RATIOS = (0.3, 0.5, 0.7, 0.85, 1.0)
REQUESTS_PER_CELL = 160
WARMUP_REQUESTS = 8
GPU_FREE_MIB = 512
PROBE_CONCURRENCY = 64
PROBE_SECONDS = 180


# Non-interactive shells do not have the project venv on PATH, and both the
# probe and run_rate_sweep.py exec `aiperf` by name.
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")


def log(message: str) -> None:
    stamp = dt.datetime.now(dt.UTC).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


class Aborted(RuntimeError):
    """Anything that must stop the whole sweep rather than produce a silent hole."""


def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, **kwargs)


def gpu_memory_used() -> dict[int, int]:
    completed = run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]
    )
    if completed.returncode != 0:
        raise Aborted(f"nvidia-smi failed: {completed.stderr.strip()}")
    used = {}
    for line in completed.stdout.strip().splitlines():
        index, memory = (part.strip() for part in line.split(","))
        used[int(index)] = int(memory)
    return used


def assert_gpus_free(gpus: list[int]) -> dict[int, int]:
    used = gpu_memory_used()
    missing = [gpu for gpu in gpus if gpu not in used]
    if missing:
        raise Aborted(f"GPUs not present on this host: {missing} (have {sorted(used)})")
    busy = {gpu: used[gpu] for gpu in gpus if used[gpu] > GPU_FREE_MIB}
    if busy:
        raise Aborted(
            f"Refusing to use GPUs held by another tenant (MiB used): {busy}. "
            "These hosts are multi-tenant."
        )
    return {gpu: used[gpu] for gpu in gpus}


def assert_ports_free(ports: list[int]) -> None:
    """These hosts are multi-tenant; a bound port is somebody else's server."""
    import socket

    taken = []
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("0.0.0.0", port))
            except OSError:
                taken.append(port)
    if taken:
        raise Aborted(f"Ports already bound on this host: {taken}. Move --base-port.")


def replica_groups(gpus: list[int], tp: int) -> list[list[int]]:
    if len(gpus) % tp:
        raise Aborted(f"{len(gpus)} GPUs do not divide into TP={tp} replicas")
    return [gpus[start : start + tp] for start in range(0, len(gpus), tp)]


def container_name(row_id: str, group: list[int]) -> str:
    return f"aws-{row_id}-gpu{group[0]}"


def stop_container(name: str) -> None:
    log(f"stopping container {name}")
    run(["docker", "stop", "-t", "30", name])
    run(["docker", "rm", "-f", name])


def docker_logs(name: str, lines: int | None = 200) -> str:
    command = ["docker", "logs", name] if lines is None else [
        "docker", "logs", "--tail", str(lines), name
    ]
    completed = run(command)
    return (completed.stdout or "") + (completed.stderr or "")


def save_docker_logs(name: str, out_dir: Path) -> None:
    """Full container log, kept before the container is removed.

    A boot failure is diagnosed from the whole log, not a tail, and `docker rm`
    destroys it; so this runs unconditionally for every replica.
    """
    directory = out_dir / "docker-logs"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.log").write_text(docker_logs(name, lines=None), encoding="utf-8")


def fork_mount_pairs(row: dict[str, Any]) -> list[tuple[Path, str]]:
    """(host source, container target) for every file of a fork-mounted row.

    A fork row replaces named vLLM runtime files in the stock image with the
    fork's versions, read-only. The list is the fork's own
    `git diff --name-only <base tag> <branch> -- vllm/`, recorded in the row, so
    the mount set cannot drift from the branch it claims to be.
    """
    spec = row.get("fork_mounts")
    if not spec:
        return []
    source_root = Path(spec["source_root"])
    target_root = spec["target_root"].rstrip("/")
    pairs = []
    for relative in spec["files"]:
        source = source_root / relative
        if not source.is_file():
            raise Aborted(
                f"Fork mount source is missing: {source}. Stage the fork worktree "
                "on this host before running a fork row; never boot one with a "
                "silently partial mount set."
            )
        pairs.append((source, f"{target_root}/{relative.removeprefix('vllm/')}"))
    return pairs


def verify_fork_mounts(row: dict[str, Any], name: str, out_dir: Path) -> None:
    """Prove the fork is the code the server is running, file by file.

    Compares the sha256 of every mounted target INSIDE the running container
    against the host source, and records a fork-only marker found in the
    container's own startup log. A row that claims a fork and cannot show both
    is refused rather than reported as a fork measurement.
    """
    pairs = fork_mount_pairs(row)
    if not pairs:
        return
    completed = run(["docker", "exec", name, "sha256sum", *[t for _, t in pairs]])
    if completed.returncode != 0:
        raise Aborted(
            f"Could not hash the mounted fork files in {name}: {completed.stderr.strip()}"
        )
    inside = {}
    for line in completed.stdout.strip().splitlines():
        digest, path = line.split()
        inside[path.lstrip("*")] = digest
    mismatched = []
    for source, target in pairs:
        expected = hashlib.sha256(source.read_bytes()).hexdigest()
        if inside.get(target) != expected:
            mismatched.append(
                {"target": target, "expected": expected, "in_container": inside.get(target)}
            )
    log_text = docker_logs(name, lines=None)
    markers = {marker: (marker in log_text) for marker in (row.get("fork_log_markers") or [])}
    receipt = {
        "container": name,
        "fork_branch": row["fork_mounts"].get("branch"),
        "fork_sha": row["fork_mounts"].get("sha"),
        "files_mounted": len(pairs),
        "sha256_matches": len(pairs) - len(mismatched),
        "mismatched": mismatched,
        "fork_only_log_markers": markers,
        "verified_utc": dt.datetime.now(dt.UTC).isoformat(),
        "how": (
            "sha256sum of every mounted target run INSIDE the running container "
            "and compared with the host source, plus a grep of the container's "
            "own startup log for a line only the fork emits."
        ),
    }
    directory = out_dir / "fork-verification"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if mismatched:
        raise Aborted(
            f"{name} does not run the fork it claims: {len(mismatched)} of "
            f"{len(pairs)} mounted files differ from the host source. {mismatched[:3]}"
        )
    missing = [marker for marker, found in markers.items() if not found]
    if missing:
        raise Aborted(
            f"{name} mounted the fork but its log carries none of the fork-only "
            f"markers {missing}; refusing to report it as a fork measurement."
        )
    log(f"{name}: fork verified - {len(pairs)} files sha256-identical, markers {markers}")


def launch_replica(row: dict[str, Any], group: list[int], port: int, out_dir: Path) -> str:
    name = container_name(row["row_id"], group)
    existing = run(["docker", "ps", "-aq", "--filter", f"name=^{name}$"]).stdout.strip()
    if existing:
        raise Aborted(
            f"A container named {name} already exists ({existing}). "
            "Remove it deliberately; this driver never reuses or force-clears names."
        )
    command = [
        "docker", "run", "-d", "--name", name,
        "--gpus", f'"device={",".join(str(gpu) for gpu in group)}"',
        "--ipc=host", "--shm-size=16g",
        "-p", f"{port}:{port}",
        "-v", "/data:/data",
    ]
    # Extra `docker run` options a row declares (2026-09-02, the NUMA arm:
    # ["--cpuset-cpus", "0-95", "--cpuset-mems", "0"]). Recorded in the
    # manifest verbatim; a row without the field gets exactly the previous
    # command.
    command.extend(row.get("docker_run_args") or [])
    for key, value in (row.get("env") or {}).items():
        command.extend(["-e", f"{key}={value}"])
    for source, target in fork_mount_pairs(row):
        command.extend(["-v", f"{source}:{target}:ro"])
    command.append(row["image"])
    command.extend([
        "--model", row["model_path"],
        "--served-model-name", row["served_model_name"],
        "--tensor-parallel-size", str(row["tp"]),
        "--port", str(port), "--host", "0.0.0.0",
    ])
    command.extend(row.get("vllm_args") or [])
    (out_dir / f"launch-{name}.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    completed = run(command)
    if completed.returncode != 0:
        # docker may have created the container before failing to publish its
        # port; remove that one by its exact name so a retry is not blocked.
        run(["docker", "rm", "-f", name])
        raise Aborted(f"docker run failed for {name}: {completed.stderr.strip()}")
    log(f"launched {name} on GPUs {group} port {port}")
    return name


def wait_healthy(name: str, port: int, timeout_s: int) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        alive = run(["docker", "inspect", "-f", "{{.State.Running}}", name]).stdout.strip()
        if alive != "true":
            raise Aborted(
                f"{name} exited before becoming healthy. Last 200 log lines:\n"
                + docker_logs(name)
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 300:
                    log(f"{name} healthy after {timeout_s - int(deadline - time.monotonic())}s")
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(5)
    raise Aborted(
        f"{name} did not report healthy within {timeout_s}s. Last 200 log lines:\n"
        + docker_logs(name)
    )


def percentiles(values: list[float], points: tuple[float, ...]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    result = {}
    for point in points:
        position = point / 100 * (len(ordered) - 1)
        low = math.floor(position)
        high = math.ceil(position)
        result[f"p{point:g}"] = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    result["mean"] = sum(ordered) / len(ordered)
    result["count"] = len(ordered)
    return result


def read_records(directory: Path) -> list[dict[str, Any]]:
    path = directory / "profile_export.jsonl"
    if not path.exists():
        raise Aborted(f"Missing per-request export: {path}")
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("metadata", {}).get("benchmark_phase") != "profiling":
            continue
        records.append(record)
    return records


def metric_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for record in records:
        entry = record.get("metrics", {}).get(key)
        if isinstance(entry, dict) and isinstance(entry.get("value"), (int, float)):
            values.append(float(entry["value"]))
    return values


def summary_value(summary: dict[str, Any], key: str) -> float | None:
    entry = summary.get(key)
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, dict):
        for candidate in ("avg", "mean", "value"):
            if isinstance(entry.get(candidate), (int, float)):
                return float(entry[candidate])
    return None


def find_cell_artifact(directory: Path) -> Path:
    """Assert the AIPerf artifact exists; a cell that produced nothing is fatal."""
    # AIPerf writes per-phase copies under phases/; the run-level export is the
    # shallowest match.
    matches = sorted(
        directory.rglob("profile_export_aiperf.json"),
        key=lambda path: len(path.relative_to(directory).parts),
    )
    if not matches:
        raise Aborted(
            f"No profile_export_aiperf.json under {directory}. The benchmark cell "
            "produced no artifact; the sweep is aborted rather than continued blind."
        )
    return matches[0]


WORKLOAD_OVERRIDE_KEYS = frozenset({"ignore_eos", "min_tokens"})


def workload_extra_inputs(row: dict[str, Any]) -> list[str]:
    """The AIPerf `--extra-inputs` this row's workload asks for.

    The campaign default forces every request to the frozen list's exact
    `output_length` with `ignore_eos:true`, so realized OSL equals nominal.
    A row's `workload_overrides` changes that: the spec-decode rows set
    `{"ignore_eos": false}` because forcing continuation past EOS degenerates
    into repetition and inflates acceptance, so they run natural lengths with
    `output_length` as `max_tokens` alone and report realized OSL beside the
    nominal. `temperature:0` is not overridable - every deliverable cell is
    greedy, and acceptance under sampling is a different number.

    A row with no `workload_overrides` gets exactly the previous behaviour.
    """
    overrides = row.get("workload_overrides") or {}
    unknown = sorted(set(overrides) - WORKLOAD_OVERRIDE_KEYS)
    if unknown:
        raise Aborted(
            f"{row['row_id']}: workload_overrides carries keys this driver does "
            f"not implement: {unknown}. Implement them or remove them; a "
            "silently ignored override would mislabel the workload."
        )
    extra_inputs = []
    if overrides.get("ignore_eos", True):
        extra_inputs.append("ignore_eos:true")
    extra_inputs.append("temperature:0")
    min_tokens = overrides.get("min_tokens")
    if min_tokens is not None:
        extra_inputs.append(f"min_tokens:{int(min_tokens)}")
    return extra_inputs


def aiperf_command(
    *,
    config: Path,
    tokenizer: str,
    base_url: str,
    extra_inputs: list[str],
    extra: list[str] | None = None,
) -> list[str]:
    command = [
        "aiperf", "profile",
        "--config", str(config),
        "--tokenizer", tokenizer,
        "--ui", "none",
        "--export-level", "records",
        "--apply-chat-template",
        "--use-server-token-count",
        "--extra-inputs", *extra_inputs,
        "--server-metrics", base_url + "/metrics",
        "--no-gpu-telemetry",
    ]
    command.extend(extra or [])
    return command


def probe_lambda_sat(
    *, row: dict[str, Any], port: int, workload_file: Path, out_dir: Path
) -> tuple[float, dict[str, Any]]:
    """Closed-loop c=64 probe on ONE replica: lambda_sat = output tok/s / mean OSL."""
    probe_dir = out_dir / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{port}"
    config = {
        "schemaVersion": "2.0",
        "random_seed": 42,
        "benchmark": {
            "model": row["served_model_name"],
            "endpoint": {
                "url": base_url + "/v1/chat/completions",
                "type": "chat",
                "streaming": True,
                "timeout": 3600.0,
            },
            "dataset": {
                "type": "file",
                "path": str(workload_file),
                "format": "single_turn",
                "sampling": "sequential",
            },
            "phases": [
                {
                    "name": "warmup",
                    "kind": "warmup",
                    "type": "concurrency",
                    "concurrency": 8,
                    "requests": WARMUP_REQUESTS,
                    "exclude_from_results": True,
                },
                {
                    "name": "profiling",
                    "kind": "profiling",
                    "type": "concurrency",
                    "concurrency": PROBE_CONCURRENCY,
                    "duration": PROBE_SECONDS,
                    "grace_period": 600,
                },
            ],
            "artifacts": {"dir": str(probe_dir), "summary": ["json"]},
        },
    }
    config_path = probe_dir / "aiperf_probe.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = aiperf_command(
        config=config_path,
        tokenizer=row["tokenizer_path"],
        base_url=base_url,
        extra_inputs=workload_extra_inputs(row),
    )
    log(f"probing lambda_sat: c={PROBE_CONCURRENCY} for {PROBE_SECONDS}s")
    completed = subprocess.run(command, cwd=str(probe_dir), text=True)
    if completed.returncode != 0:
        raise Aborted(f"lambda_sat probe failed with exit {completed.returncode}")
    summary = json.loads(find_cell_artifact(probe_dir).read_text(encoding="utf-8"))
    tok_s = summary_value(summary, "output_token_throughput")
    mean_osl = summary_value(summary, "output_sequence_length")
    if not tok_s or not mean_osl:
        raise Aborted(f"probe summary lacks throughput/OSL: {list(summary)}")
    lambda_sat = tok_s / mean_osl
    log(f"probe: {tok_s:.1f} output tok/s, mean OSL {mean_osl:.1f} -> lambda_sat {lambda_sat:.4g} req/s")
    return lambda_sat, {
        "concurrency": PROBE_CONCURRENCY,
        "duration_s": PROBE_SECONDS,
        "replica_output_tok_s": tok_s,
        "mean_osl": mean_osl,
        "lambda_sat_per_replica": lambda_sat,
    }


SPEC_DECODE_METRICS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_per_pos_total",
)


def scrape_spec_decode_counters(port: int) -> dict[str, Any]:
    """The speculative-decoding counters vLLM exposes, as one snapshot.

    Prometheus counters are monotonic since server start, so acceptance for one
    benchmark cell is the DIFFERENCE of two snapshots taken around it - the
    server has already drafted for the warmup requests and for every earlier
    cell, and the raw totals would fold those in.

    `vllm:spec_decode_num_accepted_tokens_per_pos_total` carries a `position`
    label, so it is kept as a position -> count map.
    """
    url = f"http://127.0.0.1:{port}/metrics"
    snapshot: dict[str, Any] = {
        "scraped_utc": dt.datetime.now(dt.UTC).isoformat(),
        "url": url,
    }
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            text = response.read().decode("utf-8")
    except (OSError, urllib.error.URLError) as exc:
        snapshot["error"] = f"{type(exc).__name__}: {exc}"
        return snapshot
    per_pos: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, value = line.rpartition(" ")
        name = name.strip()
        bare, _, labels = name.partition("{")
        if bare not in SPEC_DECODE_METRICS:
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if bare.endswith("per_pos_total"):
            position = "?"
            for part in labels.rstrip("}").split(","):
                key, _, raw = part.partition("=")
                if key.strip() == "position":
                    position = raw.strip().strip('"')
            per_pos[position] = per_pos.get(position, 0.0) + number
        else:
            snapshot[bare] = snapshot.get(bare, 0.0) + number
    if per_pos:
        snapshot["vllm:spec_decode_num_accepted_tokens_per_pos_total"] = per_pos
    snapshot["counters_present"] = sorted(
        key for key in snapshot if key.startswith("vllm:")
    )
    return snapshot


def spec_decode_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Acceptance over one cell: after minus before, plus the two derived rates.

    acceptance length = 1 + accepted/drafts (the mean tokens a step emits,
    counting the always-accepted bonus token); acceptance rate =
    accepted/draft_tokens (the fraction of DRAFTED tokens the target kept).
    """
    delta: dict[str, Any] = {"before": before, "after": after}
    for name in SPEC_DECODE_METRICS[:3]:
        if name in before and name in after:
            delta[name] = after[name] - before[name]
    key = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
    if isinstance(before.get(key), dict) and isinstance(after.get(key), dict):
        delta[key] = {
            position: after[key].get(position, 0.0) - count
            for position, count in before[key].items()
        }
    drafts = delta.get("vllm:spec_decode_num_drafts_total")
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens_total")
    draft_tokens = delta.get("vllm:spec_decode_num_draft_tokens_total")
    delta["acceptance_length"] = None if not drafts else 1 + accepted / drafts
    delta["acceptance_rate"] = None if not draft_tokens else accepted / draft_tokens
    return delta


def run_replica_cell(
    *,
    row: dict[str, Any],
    group: list[int],
    port: int,
    rate: float,
    workload_file: Path,
    workload_id: str,
    replica_root: Path,
    driver: Path,
) -> Path:
    # Both halves of the exact-length contract are known here: refuse a row whose
    # extras contradict the mode the frozen workload file itself carries.
    check_extra_inputs(workload_file, workload_extra_inputs(row))
    config_id = f"gpu{group[0]}"
    command = [
        sys.executable, str(driver),
        "--config-id", config_id,
        "--model", row["served_model_name"],
        "--tokenizer", row["tokenizer_path"],
        "--url", f"http://127.0.0.1:{port}/v1/chat/completions",
        "--gpu-count", str(len(group)),
        "--rates", f"{rate:.6g}",
        "--workload-file", str(workload_file),
        "--workload-id", workload_id,
        "--requests-per-rate", str(REQUESTS_PER_CELL),
        "--warmup-requests", str(WARMUP_REQUESTS),
        "--runs", "1",
        "--ui", "none",
        "--artifact-root", str(replica_root),
        "--server-container", container_name(row["row_id"], group),
        # The rate driver has its own ignore_eos:true/temperature:0 default;
        # state this row's extras explicitly so a workload_overrides row is not
        # silently benchmarked under the default it overrode.
        "--no-default-extra-inputs",
    ]
    for extra_input in workload_extra_inputs(row):
        command.extend(["--extra-input", extra_input])
    log(f"cell rate={rate:.4g} replica gpu{group[0]}: {shlex.join(command[-8:])}")
    # Only a speculative row has acceptance counters to bracket; a row that
    # declares no speculative_config gets exactly the artifacts it always did.
    wants_spec_counters = "speculative_config" in row
    before = scrape_spec_decode_counters(port) if wants_spec_counters else None
    completed = subprocess.run(command, capture_output=True, text=True)
    directory = replica_root / config_id
    (directory).mkdir(parents=True, exist_ok=True)
    if wants_spec_counters:
        (directory / "spec-decode-counters.json").write_text(
            json.dumps(
                spec_decode_delta(before, scrape_spec_decode_counters(port)), indent=2
            )
            + "\n",
            encoding="utf-8",
        )
    (directory / "wrapper.log").write_text(
        completed.stdout + "\n----- stderr -----\n" + completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise Aborted(
            f"rate cell {rate:.4g} on gpu{group[0]} exited {completed.returncode}; "
            f"see {directory / 'wrapper.log'}\n{completed.stderr[-4000:]}"
        )
    find_cell_artifact(directory)
    return directory


def aggregate_cell(
    *,
    rate: float,
    directories: list[Path],
    provisioned_gpus: int,
    measured_gpus: int,
    full_node_gpus: int,
) -> dict[str, Any]:
    per_replica = []
    ttft: list[float] = []
    itl: list[float] = []
    user_tok_s: list[float] = []
    isl: list[float] = []
    osl: list[float] = []
    node_tok_s = 0.0
    node_requests = 0
    node_errors = 0
    achieved_qps = 0.0
    spec_counters = []
    for directory in directories:
        counters_path = directory / "spec-decode-counters.json"
        if counters_path.is_file():
            spec_counters.append(json.loads(counters_path.read_text(encoding="utf-8")))
        summary = json.loads(find_cell_artifact(directory).read_text(encoding="utf-8"))
        records = read_records(find_cell_artifact(directory).parent)
        ttft.extend(metric_values(records, "time_to_first_token"))
        itl.extend(metric_values(records, "inter_token_latency"))
        user_tok_s.extend(metric_values(records, "output_token_throughput_per_user"))
        isl.extend(metric_values(records, "input_sequence_length"))
        osl.extend(metric_values(records, "output_sequence_length"))
        tok_s = summary_value(summary, "output_token_throughput") or 0.0
        node_tok_s += tok_s
        achieved_qps += summary_value(summary, "request_throughput") or 0.0
        node_requests += int(summary_value(summary, "request_count") or 0)
        errors = int(summary_value(summary, "error_request_count") or 0)
        listed = summary.get("error_summary")
        if isinstance(listed, list):
            # The export carries error_request_count only as a metric that
            # may be absent; the error_summary list is always there.
            errors = max(errors, sum(int(item.get("count", 0)) for item in listed if isinstance(item, dict)))
        node_errors += errors
        timing = rung_timing(records) if records else {}
        per_replica.append(
            {
                "artifact_dir": str(directory),
                "output_tok_s": tok_s,
                "request_throughput": summary_value(summary, "request_throughput"),
                "request_count": summary_value(summary, "request_count"),
                "error_request_count": summary_value(summary, "error_request_count"),
                "benchmark_duration_s": summary_value(summary, "benchmark_duration"),
                # From the per-request records: first dispatch to last
                # completion, first to last dispatch, and their difference,
                # which the knee test's tail correction reads.
                "records_duration_s": timing.get("duration_s"),
                "dispatch_span_s": timing.get("dispatch_span_s"),
                "drain_tail_s": timing.get("drain_tail_s"),
            }
        )
    points = (50.0, 90.0, 95.0, 99.0)
    # A row measured as ONE replica on its own GPU set, concurrently with the
    # other rows, sees full-node load while its curve stays a per-GPU curve.
    # The whole-node number is that per-GPU curve scaled by the replicas the
    # node would hold; it is PROJECTED, and labelled so, never measured.
    projection_factor = full_node_gpus / measured_gpus
    projected = {
        "measured_gpus": measured_gpus,
        "full_node_gpus": full_node_gpus,
        "node_projection_factor": projection_factor,
        "projected_full_node_output_tok_s": node_tok_s * projection_factor,
        "projected_full_node_offered_qps": rate * len(directories) * projection_factor,
        "projection_basis": (
            "measured full node"
            if projection_factor == 1
            else f"projected from {len(directories)} replica(s) under full-node load: "
            "per-replica throughput scaled by the replica count the node holds. "
            "TTFT/ITL percentiles are the measured replica's and are NOT projected."
        ),
    }
    return {
        "offered_qps_per_replica": rate,
        "offered_qps_node": rate * len(directories),
        "replicas": len(directories),
        "provisioned_gpus": provisioned_gpus,
        "node_output_tok_s": node_tok_s,
        "output_tok_s_per_provisioned_gpu": node_tok_s / provisioned_gpus,
        **projected,
        "node_achieved_qps": achieved_qps,
        "request_count": node_requests,
        "error_request_count": node_errors,
        "ttft_ms": percentiles(ttft, points),
        "itl_ms": percentiles(itl, points),
        "output_tok_s_per_user": percentiles(user_tok_s, (1.0, 50.0, 99.0)),
        "realized_isl_tokens": percentiles(isl, (50.0, 90.0, 99.0)),
        "realized_osl_tokens": percentiles(osl, (50.0, 90.0, 99.0)),
        "per_replica": per_replica,
        "spec_decode": spec_counters or None,
    }


def sweep_aggregate(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Synthesize the shape plot_aws_frontiers.py reads, at node scope."""
    combinations = []
    for cell in cells:
        combinations.append(
            {
                "parameters": {"phases.profiling.rate": cell["offered_qps_node"]},
                "metrics": {
                    "output_token_throughput": cell["node_output_tok_s"],
                    "projected_full_node_output_tok_s": cell[
                        "projected_full_node_output_tok_s"
                    ],
                    "node_projection_factor": cell["node_projection_factor"],
                    "projection_basis": cell["projection_basis"],
                    "request_throughput": cell["node_achieved_qps"],
                    "request_count": cell["request_count"],
                    "error_request_count": cell["error_request_count"],
                    "output_sequence_length": cell["realized_osl_tokens"].get("mean"),
                    "time_to_first_token": {
                        "p50": cell["ttft_ms"].get("p50"),
                        "p90": cell["ttft_ms"].get("p90"),
                        "p99": cell["ttft_ms"].get("p99"),
                    },
                    "inter_token_latency": {
                        "p50": cell["itl_ms"].get("p50"),
                        "p90": cell["itl_ms"].get("p90"),
                        "p99": cell["itl_ms"].get("p99"),
                    },
                    "output_token_throughput_per_user": {
                        "p1": cell["output_tok_s_per_user"].get("p1"),
                        "p50": cell["output_tok_s_per_user"].get("p50"),
                        "p99": cell["output_tok_s_per_user"].get("p99"),
                    },
                },
            }
        )
    return {
        "schema_version": "aws-replica-pooled/1",
        "note": (
            "Pooled over independent replicas: throughput summed, TTFT/ITL "
            "percentiles pooled over every replica's per-request records."
        ),
        "per_combination_metrics": combinations,
    }


def write_probe_only(
    *,
    out_root: Path,
    row: dict[str, Any],
    gpus: list[int],
    provisioned: int,
    full_node_gpus: int,
    lambda_sat: float,
    probe_receipt: dict[str, Any] | None,
    threshold: float,
    workload_file: Path,
    workload_id: str,
    reason: str | None = None,
) -> Path:
    """Record a row kept for its saturation probe alone, with no open-loop ladder.

    The probe is a real measurement of the deployment at c=64 and is reported
    as such; nothing here is projected onto rates that were never offered.
    """
    summary: dict[str, Any] = {}
    probe_dir = out_root / "probe"
    if probe_dir.is_dir():
        try:
            artifact = find_cell_artifact(probe_dir)
        except Aborted:
            artifact = None
        if artifact is not None:
            records = read_records(artifact.parent)
            points = (50.0, 90.0, 95.0, 99.0)
            summary = {
                "probe_ttft_ms": percentiles(metric_values(records, "time_to_first_token"), points),
                "probe_itl_ms": percentiles(metric_values(records, "inter_token_latency"), points),
                "probe_realized_isl_tokens": percentiles(
                    metric_values(records, "input_sequence_length"), (50.0, 90.0)
                ),
                "probe_realized_osl_tokens": percentiles(
                    metric_values(records, "output_sequence_length"), (50.0, 90.0)
                ),
            }
    replica_tok_s = (probe_receipt or {}).get("replica_output_tok_s")
    document = {
        "schema_version": 1,
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "status": "PROBE_ONLY",
        "config_id": row["row_id"],
        "reason": reason or (
            f"lambda_sat {lambda_sat:.4g} req/s per replica is below the "
            f"{threshold} req/s floor: the deployment is KV-bound at this "
            "topology, so the open-loop ladder was skipped and the closed-loop "
            "saturation probe is the measurement of record."
        ),
        "lambda_sat_per_replica": lambda_sat,
        "lambda_sat_floor": threshold,
        "lambda_sat_probe": probe_receipt,
        "measured_gpus": len(gpus),
        "provisioned_gpus": provisioned,
        "full_node_gpus": full_node_gpus,
        "probe_output_tok_s_per_provisioned_gpu": (
            None if replica_tok_s is None else replica_tok_s / provisioned
        ),
        "projected_full_node_output_tok_s_at_saturation": (
            None if replica_tok_s is None else replica_tok_s * (full_node_gpus / len(gpus))
        ),
        "projection_basis": (
            "closed-loop c=64 saturation probe on one replica under full-node "
            "load, scaled by full_node_gpus / measured GPUs. PROJECTED, and it "
            "is a saturation point, not a point on an offered-rate curve."
        ),
        "image": row["image"],
        "model_path": row["model_path"],
        "vllm_args": row.get("vllm_args"),
        "env": row.get("env"),
        "gpus": gpus,
        "tensor_parallel_size": row["tp"],
        "workload_id": workload_id,
        "workload_file": str(workload_file),
        "notes": row.get("notes"),
        **summary,
    }
    path = out_root / "PROBE_ONLY.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def load_existing_cells(out_root: Path) -> list[dict[str, Any]]:
    """The cells a finished row already carries, lightest first."""
    paths = sorted((out_root / "cells").glob("rate_*.json"))
    cells = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    return sorted(cells, key=lambda cell: float(cell["offered_qps_per_replica"]))


def cell_drain_tail_s(cell: dict[str, Any]) -> tuple[float | None, str]:
    """Mean drain tail over a cell's replicas, and where it came from.

    New cells carry it in per_replica; an older cell's replica artifacts are
    re-read from disk when they are still here (the node the row ran on).
    """
    tails = []
    source = "per_replica.drain_tail_s"
    for entry in cell.get("per_replica") or []:
        tail = entry.get("drain_tail_s")
        if isinstance(tail, (int, float)):
            tails.append(float(tail))
            continue
        directory = Path(entry.get("artifact_dir") or "")
        if directory.is_dir():
            try:
                tails.append(rung_timing(read_records(find_cell_artifact(directory).parent))["drain_tail_s"])
                source = "per-request records re-read from artifact_dir"
                continue
            except Aborted:
                pass
        return None, "unavailable: no drain_tail_s in the cell and no records on this host"
    if not tails:
        return None, "unavailable: cell has no replicas"
    return sum(tails) / len(tails), source


def cell_achieved_qps(
    cell: dict[str, Any], *, basis: str, tail_baseline_s: float | None
) -> tuple[float, str]:
    """The node-scope achieved request rate the knee test compares to offered."""
    if basis == "tail-corrected" and tail_baseline_s is not None:
        total = 0.0
        for entry in cell["per_replica"]:
            duration = entry.get("records_duration_s")
            count = entry.get("request_count")
            if not isinstance(duration, (int, float)) or not isinstance(count, (int, float)):
                return float(cell["node_achieved_qps"]), "raw (a replica lacks records timing)"
            total += tail_corrected_qps(
                request_count=int(count), duration_s=float(duration), tail_baseline_s=tail_baseline_s
            )
        return total, "tail-corrected"
    return float(cell["node_achieved_qps"]), "raw"


def knee_achieved_fields(verdict: dict[str, Any] | None, provisioned_gpus: int) -> dict[str, Any]:
    """What the deployment DID at the knee rung, node scope: achieved req/s (the knee
    test's own tail-corrected basis), output tok/s, tok/s per provisioned GPU, and the
    plateau (mean output tok/s over the knee rung and every rung at or above it)."""
    empty = {"knee_achieved_qps": None, "knee_output_tok_s": None,
             "knee_output_tok_s_per_provisioned_gpu": None, "plateau_output_tok_s": None,
             "plateau_output_tok_s_per_provisioned_gpu": None, "plateau_rungs": None}
    if not verdict or verdict.get("knee_ratio") is None:
        return empty
    rungs = verdict["rungs"]
    knee = next((r for r in rungs if r.get("knee") and not r.get("knee_cleared")), None)
    if knee is None:
        knee = min(rungs, key=lambda r: abs(float(r["ratio"]) - float(verdict["knee_ratio"])))
    above = [r for r in rungs if float(r["ratio"]) >= float(knee["ratio"]) - 1e-9]
    plateau = sum(float(r["output_tok_s"]) for r in above) / len(above)
    return {
        "knee_achieved_qps": float(knee["achieved_qps"]),
        "knee_output_tok_s": float(knee["output_tok_s"]),
        "knee_output_tok_s_per_provisioned_gpu": float(knee["output_tok_s"]) / provisioned_gpus,
        "plateau_output_tok_s": plateau,
        "plateau_output_tok_s_per_provisioned_gpu": plateau / provisioned_gpus,
        "plateau_rungs": [float(r["ratio"]) for r in above],
    }


def run_row(row: dict[str, Any], args: argparse.Namespace) -> Path:
    row_id = row["row_id"]
    extension = args.extend_from is not None
    stamp = args.stamp
    old_manifest: dict[str, Any] | None = None
    if extension:
        out_root = args.extend_from
        manifest_path = out_root / "aws_sweep_manifest.json"
        if not manifest_path.is_file():
            raise Aborted(f"--extend-from needs a finished row: {manifest_path} is missing")
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old_manifest.get("config_id") not in (None, row_id):
            raise Aborted(
                f"--extend-from {out_root} is row {old_manifest.get('config_id')!r}, "
                f"but --row says {row_id!r}"
            )
        cells = load_existing_cells(out_root)
        if not cells:
            raise Aborted(f"--extend-from {out_root} has no cells/rate_*.json to extend")
        receipts_dir = out_root / f"extension-{stamp}"
        receipts_dir.mkdir(parents=True, exist_ok=False)
        # The replica artifacts of the original run live beside the row on the
        # node it ran on; new rungs go there too, under their own rate_<x>/.
        candidates = [Path(old_manifest["replica_artifact_root"])] if old_manifest.get("replica_artifact_root") else []
        candidates.append(out_root.parent / f"{row_id}-replicas")
        replica_root_base = next((path for path in candidates if path.is_dir()), candidates[-1])
        replica_root_base.mkdir(parents=True, exist_ok=True)
    else:
        out_root = args.out_dir / row_id
        replica_root_base = args.out_dir / f"{row_id}-replicas"
        if out_root.exists() and any(out_root.iterdir()):
            raise Aborted(f"Output directory is not empty, write to a fresh one: {out_root}")
        out_root.mkdir(parents=True, exist_ok=True)
        replica_root_base.mkdir(parents=True, exist_ok=True)
        receipts_dir = out_root
        cells = []

    workload_file = args.workload_dir / f"requests-{row['tokenizer_id']}.jsonl"
    if not workload_file.is_file():
        raise Aborted(f"Frozen workload missing for this row's tokenizer: {workload_file}")

    gpus = list(args.gpus) if args.gpus else list(row["gpus"])
    groups = replica_groups(gpus, int(row["tp"]))
    provisioned = int(row.get("provisioned_gpus", len(gpus)))
    full_node_gpus = int(row.get("full_node_gpus", len(gpus)))
    if full_node_gpus % len(gpus):
        raise Aborted(
            f"full_node_gpus={full_node_gpus} is not a whole number of "
            f"{len(gpus)}-GPU replicas of this row"
        )
    gpu_state = assert_gpus_free(gpus)
    log(f"{row_id}: {len(groups)} replicas over GPUs {gpus} (TP={row['tp']}), "
        f"{provisioned} provisioned; pre-run MiB used {gpu_state}"
        + (f"; EXTENSION of {out_root}" if extension else ""))

    fixed_ratios = tuple(
        ratio for ratio in RATE_RATIOS
        if not any(abs(ratio - skip) < 1e-9 for skip in args.skip_ratio)
    )
    if extension:
        fixed_ratios = ()

    names: list[str] = []
    try:
        ports = [args.base_port + index for index in range(len(groups))]
        assert_ports_free(ports)
        for group, port in zip(groups, ports, strict=True):
            names.append(launch_replica(row, group, port, receipts_dir))
        for name, port in zip(names, ports, strict=True):
            wait_healthy(name, port, args.health_timeout)
            verify_fork_mounts(row, name, receipts_dir)

        probe_receipt = None
        if args.lambda_sat is not None:
            lambda_sat = float(args.lambda_sat)
            lambda_sat_source = "--lambda-sat"
        elif extension:
            lambda_sat = float(old_manifest["lambda_sat_per_replica"])
            lambda_sat_source = f"extend-from manifest ({old_manifest.get('lambda_sat_source')})"
        else:
            lambda_sat = row["lambda_sat"]
            if lambda_sat == "probe":
                if args.no_probe:
                    raise Aborted(
                        f"{row_id}: --no-probe but the row says lambda_sat 'probe' and "
                        "no --lambda-sat was given"
                    )
                lambda_sat, probe_receipt = probe_lambda_sat(
                    row=row, port=ports[0], workload_file=workload_file, out_dir=out_root
                )
                lambda_sat_source = "probe"
            elif not isinstance(lambda_sat, (int, float)):
                raise Aborted(
                    f"{row_id}: row.json states lambda_sat {lambda_sat!r}, which is neither a rate "
                    f"nor 'probe' ({row.get('lambda_sat_source')}); pass --lambda-sat"
                )
            else:
                lambda_sat_source = "row.json"
            lambda_sat = float(lambda_sat)
        log(f"{row_id}: lambda_sat {lambda_sat:.4g} req/s per replica ({lambda_sat_source})")

        if not extension and lambda_sat < args.min_lambda_sat:
            # A row this slow cannot saturate: its 0.3x cell alone would take
            # hours, and the probe already IS the finding. Keep the probe,
            # skip the ladder, and say so rather than spending the node on it.
            write_probe_only(
                out_root=out_root,
                row=row,
                gpus=gpus,
                provisioned=provisioned,
                full_node_gpus=full_node_gpus,
                lambda_sat=lambda_sat,
                probe_receipt=probe_receipt,
                threshold=args.min_lambda_sat,
                workload_file=workload_file,
                workload_id=args.workload_dir.name,
            )
            log(
                f"{row_id}: lambda_sat {lambda_sat:.4g} < --min-lambda-sat "
                f"{args.min_lambda_sat}; recorded PROBE_ONLY and skipped the ladder"
            )
            return out_root

        cells_dir = out_root / "cells"
        cells_dir.mkdir(exist_ok=True)
        new_cells: list[dict[str, Any]] = []

        def write_cell(cell: dict[str, Any]) -> None:
            (cells_dir / f"rate_{cell['offered_qps_per_replica']:.6g}.json").write_text(
                json.dumps(cell, indent=2) + "\n", encoding="utf-8"
            )

        def run_rung(ratio: float, kind: str) -> dict[str, Any]:
            rate = lambda_sat * ratio
            cell_path = cells_dir / f"rate_{rate:.6g}.json"
            if cell_path.exists():
                raise Aborted(
                    f"{row_id}: cell {cell_path} already exists; a rung is never "
                    "re-measured over an existing cell"
                )
            replica_root = replica_root_base / f"rate_{rate:.6g}"
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(groups)) as pool:
                futures = [
                    pool.submit(
                        run_replica_cell,
                        row=row,
                        group=group,
                        port=port,
                        rate=rate,
                        workload_file=workload_file,
                        workload_id=args.workload_dir.name,
                        replica_root=replica_root,
                        driver=args.driver,
                    )
                    for group, port in zip(groups, ports, strict=True)
                ]
                directories = [future.result() for future in futures]
            cell = aggregate_cell(
                rate=rate,
                directories=directories,
                provisioned_gpus=provisioned,
                measured_gpus=len(gpus),
                full_node_gpus=full_node_gpus,
            )
            cell["rate_ratio"] = ratio
            cell["lambda_sat_per_replica"] = lambda_sat
            cell["ladder"] = {"kind": kind, "run": stamp, "lambda_sat_source": lambda_sat_source}
            # The machine the rung ran on, when the row names one (a re-measurement's
            # cells may sit beside the original row's in an extension; each says whose it is).
            cell["node"] = row.get("node")
            write_cell(cell)
            cells.append(cell)
            new_cells.append(cell)
            log(
                f"{row_id} rate {rate:.4g}/replica ({ratio:.3g}x, {kind}): node "
                f"{cell['node_output_tok_s']:.1f} tok/s, "
                f"{cell['output_tok_s_per_provisioned_gpu']:.1f} tok/GPU/s, "
                f"achieved {cell['node_achieved_qps']:.3g}/{cell['offered_qps_node']:.3g} req/s raw, "
                f"TTFT p99 {cell['ttft_ms'].get('p99', float('nan')):.0f} ms, "
                f"ITL p99 {cell['itl_ms'].get('p99', float('nan')):.1f} ms, "
                f"errors {cell['error_request_count']}"
            )
            return cell

        for ratio in fixed_ratios:
            run_rung(ratio, "fixed")

        verdict = None
        knee_basis = None
        tail_baseline: tuple[float | None, str] = (None, "not needed")
        if args.until_knee:
            if not cells:
                raise Aborted(f"{row_id}: --until-knee with no rung below the ladder to start from")
            lightest = min(cells, key=lambda cell: float(cell["offered_qps_per_replica"]))
            heaviest = max(cells, key=lambda cell: float(cell["offered_qps_per_replica"]))
            prior_tok_s = float(heaviest["node_output_tok_s"])
            if args.knee_achieved == "tail-corrected":
                tail_baseline = cell_drain_tail_s(lightest)
                if tail_baseline[0] is None:
                    log(
                        f"{row_id}: WARNING tail-corrected knee test has no baseline "
                        f"({tail_baseline[1]}); falling back to AIPerf's raw request_throughput"
                    )
            log(
                f"{row_id}: ladder from {args.start_ratio}x, prior tok/s {prior_tok_s:.1f} "
                f"({heaviest['rate_ratio']}x), tail baseline {tail_baseline[0]} s "
                f"({tail_baseline[1]} at {lightest['rate_ratio']}x)"
            )

            def measure(rate: float) -> tuple[float, float, int]:
                ratio = round(rate / lambda_sat, 6)
                cell = run_rung(ratio, "ladder")
                achieved, basis = cell_achieved_qps(
                    cell, basis=args.knee_achieved, tail_baseline_s=tail_baseline[0]
                )
                cell["ladder"].update(
                    {
                        "achieved_qps": achieved,
                        "achieved_fraction": achieved / float(cell["offered_qps_node"]),
                        "achieved_basis": basis,
                        "tail_baseline_s": tail_baseline[0],
                        "tail_baseline_source": tail_baseline[1],
                    }
                )
                write_cell(cell)
                log(
                    f"{row_id} rung {ratio:.3g}x: achieved {achieved:.3g} req/s = "
                    f"{cell['ladder']['achieved_fraction']:.3f} of offered ({basis})"
                )
                return achieved, float(cell["node_output_tok_s"]), int(cell["error_request_count"])

            verdict = knee_ladder(
                measure,
                lambda_sat=lambda_sat,
                start_ratio=args.start_ratio,
                step=args.ladder_step,
                knee_fraction=args.knee_fraction,
                plateau_tolerance=args.plateau_tolerance,
                confirmation_factor=args.confirmation_factor,
                prior_output_tok_s=prior_tok_s,
                max_rungs=args.max_ladder_rungs,
            )
            ladder_cells = [cell for cell in new_cells if cell["ladder"]["kind"] == "ladder"]
            for cell, rung in zip(ladder_cells, verdict["rungs"], strict=True):
                cell["ladder"]["kind"] = rung["kind"]
                cell["ladder"]["knee"] = rung["knee"]
                for key in ("knee_cleared", "confirms_ratio", "gain_vs_previous", "gain_vs_knee"):
                    if key in rung:
                        cell["ladder"][key] = rung[key]
                write_cell(cell)
            knee_basis = next(
                (cell["ladder"]["achieved_basis"] for cell in ladder_cells), args.knee_achieved
            )
            log(
                f"{row_id}: ladder stopped ({verdict['stop_reason']}) after "
                f"{verdict['rungs_run']} rungs; knee_rate {verdict['knee_rate']} "
                f"(ratio {verdict['knee_ratio']}, confirmed {verdict['knee_confirmed']})"
            )

        cells.sort(key=lambda cell: float(cell["offered_qps_per_replica"]))
        (out_root / "sweep_aggregate").mkdir(exist_ok=True)
        (out_root / "sweep_aggregate" / "profile_export_aiperf_sweep.json").write_text(
            json.dumps(sweep_aggregate(cells), indent=2) + "\n", encoding="utf-8"
        )
        knee_fields = {
            "until_knee": bool(args.until_knee),
            "knee_rate": None if verdict is None else verdict["knee_rate"],
            "knee_ratio": None if verdict is None else verdict["knee_ratio"],
            "knee_confirmed": None if verdict is None else verdict["knee_confirmed"],
            "stop_reason": None if verdict is None else verdict["stop_reason"],
            "rungs_run": None if verdict is None else verdict["rungs_run"],
            "knee_achieved_basis": knee_basis,
            "tail_baseline_s": tail_baseline[0],
            "tail_baseline_source": tail_baseline[1],
            "ladder": verdict,
            # The figures quote the ACHIEVED plateau at the knee rung, not the offered
            # knee_rate: the offered rate at the knee overstates a
            # deployment whose achieved req/s stopped growing rungs earlier.
            **knee_achieved_fields(verdict, provisioned),
        }
        if extension:
            assert old_manifest is not None
            backup = out_root / f"aws_sweep_manifest.pre-extension-{stamp}.json"
            backup.write_text(json.dumps(old_manifest, indent=2) + "\n", encoding="utf-8")
            manifest = dict(old_manifest)
            manifest["status"] = "COMPLETE"
            manifest["extended_utc"] = dt.datetime.now(dt.UTC).isoformat()
            manifest["rate_ratios"] = sorted(
                {float(r) for r in manifest.get("rate_ratios") or []}
                | {float(cell["rate_ratio"]) for cell in cells}
            )
            manifest.update(knee_fields)
            manifest.setdefault("extensions", []).append(
                {
                    "run": stamp,
                    "node": row.get("node"),
                    "receipts_dir": str(receipts_dir),
                    "gpus": gpus,
                    "ports": ports,
                    "lambda_sat_per_replica": lambda_sat,
                    "lambda_sat_source": lambda_sat_source,
                    "rate_ratios": [cell["rate_ratio"] for cell in new_cells],
                    "cells_written": [f"rate_{cell['offered_qps_per_replica']:.6g}.json" for cell in new_cells],
                    "previous_manifest": str(backup),
                }
            )
        else:
            manifest = {
                "schema_version": 1,
                "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
                "status": "COMPLETE",
                "config_id": row_id,
                "node": row.get("node"),
                "model": row["served_model_name"],
                "tokenizer": row["tokenizer_path"],
                "gpu_count": provisioned,
                "replicas": len(groups),
                "tensor_parallel_size": row["tp"],
                "gpus": gpus,
                "image": row["image"],
                "model_path": row["model_path"],
                "vllm_args": row.get("vllm_args"),
                "env": row.get("env"),
                "fork_mounts": row.get("fork_mounts"),
                "docker_run_args": row.get("docker_run_args"),
                "full_node_gpus": full_node_gpus,
                "node_projection_factor": full_node_gpus / len(gpus),
                "workload_id": args.workload_dir.name,
                "workload_file": str(workload_file),
                "workload_overrides": row.get("workload_overrides"),
                "extra_inputs": workload_extra_inputs(row),
                "requests_per_cell_per_replica": REQUESTS_PER_CELL,
                "warmup_requests": WARMUP_REQUESTS,
                "requests_note": row.get("requests_note"),
                "rate_ratios": [cell["rate_ratio"] for cell in cells],
                "fixed_rate_ratios": list(fixed_ratios),
                "skipped_rate_ratios": list(args.skip_ratio),
                "lambda_sat_per_replica": lambda_sat,
                "lambda_sat_probe": probe_receipt,
                "lambda_sat_source": lambda_sat_source,
                "replica_artifact_root": str(replica_root_base),
                "notes": row.get("notes"),
                **knee_fields,
            }
        (out_root / "aws_sweep_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return out_root
    finally:
        for name in names:
            save_docker_logs(name, receipts_dir)
            stop_container(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--row", action="append", type=Path,
                        help="Repeatable: path to a deployments/<id>/row.json "
                             "(with --extend-from, defaults to <dir>/row.json)")
    parser.add_argument("--out-dir", type=Path,
                        help="Batch directory the row directories go under (not with --extend-from)")
    parser.add_argument("--workload-dir", type=Path, required=True,
                        help="workload/aws-p50p90-v1")
    parser.add_argument("--driver", type=Path,
                        default=Path(__file__).resolve().parent / "run_rate_sweep.py")
    parser.add_argument("--base-port", type=int, default=8100)
    parser.add_argument(
        "--min-lambda-sat",
        type=float,
        default=0.0,
        help=(
            "Saturation-rate floor in req/s per replica. A probe below it records "
            "PROBE_ONLY and skips the open-loop ladder, whose lowest cell would "
            "otherwise run for hours."
        ),
    )
    parser.add_argument(
        "--marker-suffix",
        default="",
        help=(
            "Appended to the DONE/ABORTED marker names. Rows of one batch run "
            "as separate parallel invocations sharing --out-dir; without a "
            "per-invocation suffix they would overwrite each other's marker."
        ),
    )
    parser.add_argument("--health-timeout", type=int, default=1800)
    parser.add_argument(
        "--requests-per-rate", type=int, default=REQUESTS_PER_CELL,
        help=(
            "Requests per rung per replica (default 160, the frozen list once over). "
            "A rung's in-flight count can never exceed its request count, so a "
            "max-num-seqs >= 128 deployment needs more than 160 to reach its "
            "plateau; a count above the workload file's line count cycles the "
            "same prompts in file order (AIPerf sequential sampling)."
        ),
    )
    parser.add_argument("--warmup-requests", type=int, default=WARMUP_REQUESTS)
    parser.add_argument(
        "--probe-concurrency",
        type=int,
        default=PROBE_CONCURRENCY,
        help=(
            "Closed-loop concurrency of the lambda_sat probe. The default is the "
            "campaign's c=64; a deployment left at vLLM's default max_num_seqs of "
            "256 can be under-saturated at 64, which would put the ladder's 1.0x "
            "cell below the real knee."
        ),
    )
    parser.add_argument(
        "--rate-ratios",
        help=(
            "Comma-separated multiples of lambda_sat to sweep; defaults to "
            + ",".join(str(ratio) for ratio in RATE_RATIOS)
        ),
    )

    knee = parser.add_argument_group("knee-finding ladder (module docstring)")
    knee.add_argument("--until-knee", action="store_true")
    knee.add_argument("--no-probe", action="store_true")
    knee.add_argument("--lambda-sat", type=float)
    knee.add_argument("--extend-from", type=Path, metavar="ROW_DIR")
    knee.add_argument("--skip-ratio", type=float, action="append", default=[], metavar="RATIO")
    knee.add_argument("--start-ratio", type=float, default=LADDER_START_RATIO)
    knee.add_argument("--ladder-step", type=float, default=LADDER_STEP)
    knee.add_argument("--knee-fraction", type=float, default=KNEE_ACHIEVED_FRACTION)
    knee.add_argument("--plateau-tolerance", type=float, default=PLATEAU_TOLERANCE)
    knee.add_argument("--confirmation-factor", type=float, default=CONFIRMATION_FACTOR)
    knee.add_argument("--max-ladder-rungs", type=int, default=MAX_LADDER_RUNGS)
    knee.add_argument("--knee-achieved", choices=KNEE_ACHIEVED_BASES, default="tail-corrected")
    knee.add_argument("--gpus", help="Comma-separated GPU indices overriding the row's")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rate_ratios:
        global RATE_RATIOS
        RATE_RATIOS = tuple(float(part) for part in args.rate_ratios.split(",") if part.strip())
    global PROBE_CONCURRENCY
    PROBE_CONCURRENCY = args.probe_concurrency
    global REQUESTS_PER_CELL, WARMUP_REQUESTS
    REQUESTS_PER_CELL = args.requests_per_rate
    WARMUP_REQUESTS = args.warmup_requests
    if REQUESTS_PER_CELL < 1 or WARMUP_REQUESTS < 1:
        raise SystemExit("--requests-per-rate and --warmup-requests must be >= 1")
    args.stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    args.gpus = [int(part) for part in args.gpus.split(",") if part.strip()] if args.gpus else None
    if args.lambda_sat is not None and args.lambda_sat <= 0:
        raise SystemExit("--lambda-sat must be > 0")
    args.workload_dir = args.workload_dir.resolve()

    if args.extend_from is not None:
        args.extend_from = args.extend_from.resolve()
        if not args.extend_from.is_dir():
            raise SystemExit(f"--extend-from is not a directory: {args.extend_from}")
        if args.out_dir is not None:
            raise SystemExit("--extend-from writes into the row directory itself; drop --out-dir")
        if not args.until_knee:
            raise SystemExit("--extend-from runs only the knee ladder; pass --until-knee")
        if args.skip_ratio:
            raise SystemExit("--skip-ratio applies to the fixed ratios, which an extension does not run")
        rows = args.row or [args.extend_from / "row.json"]
        if len(rows) != 1:
            raise SystemExit("--extend-from extends exactly one row")
        if not rows[0].is_file():
            raise SystemExit(f"No row.json for the extension: pass --row (looked at {rows[0]})")
        args.row = rows
        args.out_dir = args.extend_from.parent
        done = args.extend_from / f"EXTENSION-DONE-{args.stamp}"
        aborted = args.extend_from / f"EXTENSION-ABORTED-{args.stamp}"
    else:
        if not args.row or args.out_dir is None:
            raise SystemExit("--row and --out-dir are required (or --extend-from)")
        args.out_dir = args.out_dir.resolve()
        args.out_dir.mkdir(parents=True, exist_ok=True)
        done = args.out_dir / f"DONE{args.marker_suffix}"
        aborted = args.out_dir / f"ABORTED{args.marker_suffix}"
        if done.exists() or aborted.exists():
            raise SystemExit(f"{args.out_dir} already carries a terminal marker; use a fresh directory")

    finished = []
    for row_path in args.row:
        row = json.loads(row_path.read_text(encoding="utf-8"))
        log(f"=== row {row['row_id']} from {row_path}")
        try:
            finished.append(str(run_row(row, args)))
        except BaseException as exc:  # marker first, then re-raise
            aborted.write_text(
                json.dumps(
                    {
                        "aborted_utc": dt.datetime.now(dt.UTC).isoformat(),
                        "row": row.get("row_id"),
                        "row_file": str(row_path),
                        "completed_rows": finished,
                        "reason": f"{type(exc).__name__}: {exc}",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            log(f"ABORTED: {type(exc).__name__}: {exc}")
            raise
    done.write_text(
        json.dumps(
            {
                "done_utc": dt.datetime.now(dt.UTC).isoformat(),
                "rows": finished,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"DONE: {len(finished)} rows -> {done}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
