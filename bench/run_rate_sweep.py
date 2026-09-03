#!/usr/bin/env python3
"""Generate and optionally run a finite, open-loop AIPerf request-rate sweep.

The serving deployment is assumed to already be running and fixed. This script
varies only the offered request rate. By default it deliberately omits a
concurrency limit, so AIPerf does not turn overload into a client-side
closed-loop/semaphore-limited workload.

Designed for exactly AIPerf 0.12.0 and Python 3.11+.

Knee-finding ladder (2026-09-02, "Saturation and re-run plan", section a):

    --until-knee --lambda-sat <req/s> [--start-ratio 1.1] [--rates ...]

runs the given --rates first (may be omitted for an extension run), one
AIPerf run per rung, then steps +0.1 x lambda_sat from --start-ratio (1.1,
1.2, 1.3, ...) and evaluates after every rung:

  * knee: the first rung whose achieved request rate is below 95 % of the
    offered rate. At the knee ONE confirmation rung runs at 1.5 x the knee
    rate. If its output tok/s is within 3 % of the knee rung's, the ladder
    stops (stop_reason "plateau"); if it is higher, the knee label was
    premature: it is cleared and +0.1 stepping resumes from the confirmation
    rung's ratio.
  * plateau-before-knee: two consecutive +0.1 rungs that each gain < 3 %
    output tok/s stop the ladder without a knee.
  * errors: any errored request in a rung stops the ladder (a broken
    measurement, not a data point).

"Achieved request rate" defaults to --knee-achieved tail-corrected: the
rung's request_count divided by (benchmark duration - drain tail of the
lightest rung), where the drain tail is the time from the last dispatch to
the last completion, read from the per-request records. AIPerf's raw
`request_throughput` = requests / benchmark_duration includes that tail, and
on the campaign's finite 160-request rungs the tail alone puts the raw ratio
at 0.78-0.99 at 1.0 x lambda_sat on every existing row, so the raw rule would
label the first ladder rung the knee on nearly every deployment.
`--knee-achieved raw` selects the literal AIPerf field.

Every rung's export lands in rate_<x>__rate_<x>/profile_export_aiperf.json
(AIPerf's own sweep layout), and aws_sweep_manifest.json gains knee_rate,
knee_ratio, stop_reason, rungs_run and a per-rung table. The ladder logic is
the pure function `knee_ladder`, tested in scripts/tests/test_until_knee.py
without AIPerf. Without --until-knee nothing here changes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import random
import re
import shlex
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from statistics import NormalDist
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - user-facing dependency error
    raise SystemExit(
        "PyYAML is required. Install the project dependencies with: uv sync"
    ) from exc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workload_variant import check_extra_inputs  # noqa: E402  (sibling script)


AIPERF_VERSION = "0.12.0"
DEFAULT_RATE_RATIOS = "0.15,0.30,0.45,0.60,0.72,0.82,0.90,0.96,1.00,1.05"

# The knee-finding ladder (module docstring). Ratios are multiples of lambda_sat.
LADDER_START_RATIO = 1.1
LADDER_STEP = 0.1
KNEE_ACHIEVED_FRACTION = 0.95
PLATEAU_TOLERANCE = 0.03
CONFIRMATION_FACTOR = 1.5
MAX_LADDER_RUNGS = 40
KNEE_ACHIEVED_BASES = ("tail-corrected", "raw")


def knee_ladder(
    run_rung,
    *,
    lambda_sat: float,
    start_ratio: float = LADDER_START_RATIO,
    step: float = LADDER_STEP,
    knee_fraction: float = KNEE_ACHIEVED_FRACTION,
    plateau_tolerance: float = PLATEAU_TOLERANCE,
    confirmation_factor: float = CONFIRMATION_FACTOR,
    prior_output_tok_s: float | None = None,
    max_rungs: int = MAX_LADDER_RUNGS,
) -> dict[str, Any]:
    """Drive the +0.1 x lambda_sat ladder to its knee; pure, no I/O.

    `run_rung(rate) -> (achieved_qps, output_tok_s, errors)` measures one rung
    at an offered rate in req/s. `prior_output_tok_s` seeds the gain test with
    the rung just below the ladder (the last fixed rate, or the highest cell an
    extension run continues from); without it the first ladder rung has no
    predecessor and the gain test starts at the second.

    Returns the rung table and the verdict: knee_rate/knee_ratio (None when
    the ladder stopped before a knee), knee_confirmed, stop_reason in
    {"plateau", "plateau-before-knee", "errors", "max-rungs"}, rungs_run.
    """
    if lambda_sat <= 0:
        raise ValueError("lambda_sat must be > 0")
    if step <= 0 or start_ratio <= 0:
        raise ValueError("start_ratio and step must be > 0")
    rungs: list[dict[str, Any]] = []
    knee: dict[str, Any] | None = None
    knee_confirmed = False
    stop_reason: str | None = None
    prev_tok_s = prior_output_tok_s
    small_gains = 0
    ratio = round(start_ratio, 6)

    def measure(ratio: float, kind: str) -> dict[str, Any]:
        rate = ratio * lambda_sat
        achieved_qps, output_tok_s, errors = run_rung(rate)
        entry = {
            "index": len(rungs),
            "ratio": ratio,
            "rate": rate,
            "kind": kind,
            "achieved_qps": float(achieved_qps),
            "achieved_fraction": float(achieved_qps) / rate,
            "output_tok_s": float(output_tok_s),
            "errors": int(errors),
            "knee": False,
        }
        if prev_tok_s:
            entry["gain_vs_previous"] = float(output_tok_s) / prev_tok_s - 1
        rungs.append(entry)
        return entry

    while True:
        if len(rungs) >= max_rungs:
            stop_reason = "max-rungs"
            break
        entry = measure(ratio, "step")
        if entry["errors"]:
            stop_reason = "errors"
            break
        if entry["achieved_qps"] < knee_fraction * entry["rate"]:
            entry["knee"] = True
            knee = entry
            confirmation = measure(round(ratio * confirmation_factor, 6), "confirmation")
            confirmation["confirms_ratio"] = ratio
            if confirmation["errors"]:
                stop_reason = "errors"
                break
            confirmation["gain_vs_knee"] = (
                confirmation["output_tok_s"] / knee["output_tok_s"] - 1
                if knee["output_tok_s"]
                else None
            )
            if confirmation["output_tok_s"] <= knee["output_tok_s"] * (1 + plateau_tolerance):
                knee_confirmed = True
                stop_reason = "plateau"
                break
            # The confirmation rung produced more: the knee label was premature.
            entry["knee"] = False
            entry["knee_cleared"] = True
            knee = None
            prev_tok_s = confirmation["output_tok_s"]
            small_gains = 0
            ratio = round(confirmation["ratio"] + step, 6)
            continue
        if prev_tok_s:
            if entry["output_tok_s"] < prev_tok_s * (1 + plateau_tolerance):
                small_gains += 1
            else:
                small_gains = 0
            if small_gains >= 2:
                stop_reason = "plateau-before-knee"
                break
        prev_tok_s = entry["output_tok_s"]
        ratio = round(ratio + step, 6)

    return {
        "parameters": {
            "lambda_sat": lambda_sat,
            "start_ratio": start_ratio,
            "step": step,
            "knee_fraction": knee_fraction,
            "plateau_tolerance": plateau_tolerance,
            "confirmation_factor": confirmation_factor,
            "prior_output_tok_s": prior_output_tok_s,
            "max_rungs": max_rungs,
        },
        "rungs": rungs,
        "rungs_run": len(rungs),
        "knee_rate": None if knee is None else knee["rate"],
        "knee_ratio": None if knee is None else knee["ratio"],
        "knee_confirmed": knee_confirmed,
        "stop_reason": stop_reason,
        "stop_ratio": rungs[-1]["ratio"] if rungs else None,
    }


def profiling_records(export_dir: Path) -> list[dict[str, Any]]:
    """The profiling-phase per-request records of one AIPerf run."""
    path = export_dir / "profile_export.jsonl"
    if not path.is_file():
        raise SystemExit(f"Missing per-request export: {path}")
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("metadata", {}).get("benchmark_phase") == "profiling":
            records.append(record)
    return records


def rung_timing(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Dispatch span and drain tail of one rung, from its per-request records.

    duration = first dispatch to last completion (what AIPerf divides by);
    dispatch_span = first to last dispatch; drain_tail = duration - span, the
    time the last-dispatched requests take to finish. Below the knee the tail
    is one request's unloaded latency; above it, the queue those requests wait
    in - the growth of the tail is what the tail-corrected knee test sees.
    """
    starts = [
        int(record["metadata"]["request_start_ns"])
        for record in records
        if record.get("metadata", {}).get("request_start_ns") is not None
    ]
    ends = [
        int(record["metadata"]["request_end_ns"])
        for record in records
        if record.get("metadata", {}).get("request_end_ns") is not None
    ]
    if not starts or not ends:
        raise SystemExit("Per-request records carry no request_start_ns/request_end_ns")
    duration_s = (max(ends) - min(starts)) / 1e9
    span_s = (max(starts) - min(starts)) / 1e9
    return {
        "request_count": len(records),
        "duration_s": duration_s,
        "dispatch_span_s": span_s,
        "drain_tail_s": duration_s - span_s,
    }


def tail_corrected_qps(
    *, request_count: int, duration_s: float, tail_baseline_s: float
) -> float:
    """Requests per second with the lightest rung's drain tail taken off.

    With a fixed seed the arrival pattern is identical at every rung, so the
    only part of the duration that changes with load is the tail; subtracting
    the unloaded tail leaves the dispatch span plus the queueing excess.
    """
    effective = max(duration_s - tail_baseline_s, 1e-9)
    return request_count / effective


def export_error_count(summary: dict[str, Any]) -> int:
    """Errored requests in one export: the metric when present, else error_summary."""
    count = 0
    entry = summary.get("error_request_count")
    if isinstance(entry, (int, float)):
        count = int(entry)
    elif isinstance(entry, dict):
        for candidate in ("avg", "value", "sum"):
            if isinstance(entry.get(candidate), (int, float)):
                count = int(entry[candidate])
                break
    listed = summary.get("error_summary")
    if isinstance(listed, list):
        count = max(count, sum(int(item.get("count", 0)) for item in listed if isinstance(item, dict)))
    return count


def export_metric(summary: dict[str, Any], key: str) -> float | None:
    entry = summary.get(key)
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, dict):
        for candidate in ("avg", "mean", "value"):
            if isinstance(entry.get(candidate), (int, float)):
                return float(entry[candidate])
    return None


def parse_csv_floats(value: str, *, name: str) -> list[float]:
    values: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            number = float(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid {name} value: {part!r}") from exc
        if not math.isfinite(number) or number <= 0:
            raise argparse.ArgumentTypeError(
                f"{name} values must be finite and > 0: {part!r}"
            )
        values.append(number)
    if not values:
        raise argparse.ArgumentTypeError(f"At least one {name} value is required")
    return values


def stable_unique_sorted(values: Iterable[float]) -> list[float]:
    normalized = {float(f"{value:.10g}") for value in values}
    return sorted(normalized)


def bounded_lognormal(*, p50: float, p90: float, cap: int) -> dict[str, Any]:
    """Return AIPerf's mean/median parameterization plus an audit receipt."""
    if not 0 < p50 < p90 < cap:
        raise ValueError(f"Require 0 < p50 < p90 < cap, got {p50}, {p90}, {cap}")
    sigma = math.log(p90 / p50) / NormalDist().inv_cdf(0.9)
    mean = p50 * math.exp((sigma * sigma) / 2)
    return {
        "config": {
            "type": "lognormal",
            "mean": mean,
            "median": p50,
            "min": 1,
            "max": cap,
        },
        "receipt": {
            "family": "bounded_lognormal",
            "p50": p50,
            "p90": p90,
            "sigma": sigma,
            "unbounded_mean": mean,
            "cap": cap,
        },
    }


def quantile_values(*, p50: float, p90: float, cap: int, count: int = 100) -> list[int]:
    """Build a deterministic bounded-lognormal quantile table.

    AIPerf 0.12 validates scalar sampling-distribution objects on prompts.isl/osl
    but its synthetic composer consumes only their expected value. The older
    sequence_distribution path samples correctly, so represent each marginal
    as 100 equal-weight fixed quantiles and pair them with a seeded permutation.
    """
    receipt = bounded_lognormal(p50=p50, p90=p90, cap=cap)["receipt"]
    sigma = float(receipt["sigma"])
    values = []
    for index in range(count):
        probability = (index + 0.5) / count
        value = math.exp(math.log(p50) + sigma * NormalDist().inv_cdf(probability))
        values.append(max(1, min(cap, round(value))))
    # Make nearest-rank and linearly interpolated p50/p90 exact for N=100.
    if count == 100:
        values[49] = values[50] = round(p50)
        values[89] = values[90] = round(p90)
    return sorted(values)


def empirical_sequence_distribution(args: argparse.Namespace) -> list[dict[str, Any]]:
    isl_values = [
        max(1, value - args.chat_template_overhead_tokens)
        for value in quantile_values(
            p50=args.isl_p50, p90=args.isl_p90, cap=args.isl_cap
        )
    ]
    osl_values = quantile_values(p50=args.osl_p50, p90=args.osl_p90, cap=args.osl_cap)
    random.Random(args.seed + 1).shuffle(osl_values)
    return [
        {"isl": isl, "osl": osl, "probability": 1.0}
        for isl, osl in zip(isl_values, osl_values, strict=True)
    ]


def compute_rates(args: argparse.Namespace) -> tuple[list[float], float | None]:
    if args.rates:
        rates = stable_unique_sorted(parse_csv_floats(args.rates, name="rate"))
        return rates, None

    if args.until_knee and args.capacity_output_tok_s is None:
        # An extension run: no fixed rungs, the ladder alone.
        return [], None

    if args.capacity_output_tok_s is None:
        raise SystemExit(
            "Provide either --rates or --capacity-output-tok-s. "
            "The latter is converted to approximate saturation QPS using the "
            "surrogate distribution's unbounded mean OSL."
        )

    osl = bounded_lognormal(p50=args.osl_p50, p90=args.osl_p90, cap=args.osl_cap)
    mean_osl = float(osl["receipt"]["unbounded_mean"])
    estimated_saturation_qps = args.capacity_output_tok_s / mean_osl
    ratios = parse_csv_floats(args.rate_ratios, name="rate ratio")
    rates = stable_unique_sorted(estimated_saturation_qps * ratio for ratio in ratios)
    return rates, estimated_saturation_qps


def compute_durations(
    rates: list[float],
    *,
    target_requests: int,
    min_duration: int,
    max_duration: int,
) -> list[int]:
    if target_requests < 1:
        raise ValueError("target_requests must be >= 1")
    if min_duration < 1 or max_duration < min_duration:
        raise ValueError("Require 1 <= min_duration <= max_duration")
    durations = []
    for rate in rates:
        desired = math.ceil(target_requests / rate)
        durations.append(max(min_duration, min(max_duration, desired)))
    return durations


def normalize_chat_url(value: str) -> tuple[str, str]:
    if "://" not in value:
        value = "http://" + value
    parsed = urllib.parse.urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        raise SystemExit(f"Invalid endpoint URL: {value!r}")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1/chat/completions"):
        base_path = path[: -len("/v1/chat/completions")]
        chat_path = path
    elif path.endswith("/v1"):
        base_path = path[: -len("/v1")]
        chat_path = path + "/chat/completions"
    elif not path:
        base_path = ""
        chat_path = "/v1/chat/completions"
    else:
        raise SystemExit(
            "--url must be a server base URL, a /v1 URL, or a full "
            "/v1/chat/completions URL"
        )
    base_url = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, base_path, "", "")
    ).rstrip("/")
    chat_url = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, chat_path, "", "")
    )
    return base_url, chat_url


def get_json(url: str, timeout: float = 15) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if not 200 <= response.status < 300:
                raise SystemExit(f"GET {url} returned HTTP {response.status}")
            value = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise SystemExit(f"GET {url} failed: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"GET {url} did not return a JSON object")
    return value


def validate_endpoint(base_url: str, model: str, maximum_total_tokens: int) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(base_url + "/health", timeout=15) as response:
            if not 200 <= response.status < 300:
                raise SystemExit(f"Health endpoint returned HTTP {response.status}")
    except (OSError, urllib.error.URLError) as exc:
        raise SystemExit(f"Health check failed at {base_url}/health: {exc}") from exc

    payload = get_json(base_url + "/v1/models")
    models = payload.get("data")
    if not isinstance(models, list):
        raise SystemExit("/v1/models response has no data list")
    selected = next(
        (entry for entry in models if isinstance(entry, dict) and entry.get("id") == model),
        None,
    )
    if selected is None:
        ids = [entry.get("id") for entry in models if isinstance(entry, dict)]
        raise SystemExit(f"Model {model!r} is absent from /v1/models: {ids}")
    max_model_len = selected.get("max_model_len")
    if not isinstance(max_model_len, int):
        raise SystemExit("/v1/models does not report an integer max_model_len")
    if maximum_total_tokens > max_model_len:
        raise SystemExit(
            f"Workload cap {maximum_total_tokens} exceeds max_model_len {max_model_len}"
        )
    return {
        "health": "HTTP 2xx",
        "served_model": model,
        "max_model_len": max_model_len,
        "maximum_workload_tokens": maximum_total_tokens,
    }


def collect_container_receipt(container: str | None) -> dict[str, Any] | None:
    if container is None:
        return None
    if shutil.which("docker") is None:
        raise SystemExit("--server-container was supplied but docker is unavailable")
    inspected = subprocess.run(
        ["docker", "inspect", container],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(inspected.stdout)[0]
    args = data.get("Args", [])
    if "--no-enable-prefix-caching" not in args:
        raise SystemExit(
            f"Container {container!r} does not have --no-enable-prefix-caching"
        )
    return {
        "container": container,
        "container_id": data.get("Id"),
        "image": data.get("Image"),
        "args": args,
        "prefix_caching_verified_disabled": True,
    }


def build_config(
    args: argparse.Namespace,
    rates: list[float],
    durations: list[int],
    artifact_root: Path,
    chat_url: str,
) -> dict[str, Any]:
    warmup_phase: dict[str, Any] = {
        "name": "warmup",
        "kind": "warmup",
        "type": args.arrival_pattern,
        "rate": rates[0],
        "exclude_from_results": True,
    }
    if args.warmup_requests is not None:
        warmup_phase["requests"] = args.warmup_requests
    else:
        warmup_phase["duration"] = args.warmup_duration
        warmup_phase["grace_period"] = args.warmup_grace_period

    profiling_phase: dict[str, Any] = {
        "name": "profiling",
        "kind": "profiling",
        "type": args.arrival_pattern,
        "rate": rates[0],
    }
    if args.requests_per_rate is not None:
        profiling_phase["requests"] = args.requests_per_rate
    else:
        profiling_phase["duration"] = durations[0]
        profiling_phase["grace_period"] = args.grace_period

    if args.max_concurrency is not None:
        warmup_phase["concurrency"] = args.max_concurrency
        profiling_phase["concurrency"] = args.max_concurrency

    if args.workload_file is not None:
        # The frozen list: every cell and every deployment replays these exact
        # (prompt, output_length) records, in order, instead of regenerating a
        # sample of the surrogate distribution.
        dataset: dict[str, Any] = {
            "type": "file",
            "path": str(args.workload_file),
            "format": "single_turn",
            "sampling": "sequential",
        }
    else:
        dataset = {
            "type": "synthetic",
            "entries": args.dataset_entries,
            "prompts": {
                "corpus": args.prompt_corpus,
                # AIPerf 0.12's synthetic enablement check requires positive
                # scalar fallbacks even when sequence_distribution overrides them.
                "isl": 1,
                "osl": 1,
                "sequence_distribution": empirical_sequence_distribution(args),
            },
        }

    return {
        "schemaVersion": "2.0",
        "random_seed": args.seed,
        "benchmark": {
            "model": args.model,
            "endpoint": {
                "url": chat_url,
                "type": "chat",
                "streaming": True,
                "timeout": args.endpoint_timeout,
            },
            "dataset": dataset,
            "phases": [warmup_phase, profiling_phase],
            "artifacts": {
                "dir": str(artifact_root),
                "summary": ["json"],
            },
        },
        "sweep": {
            "type": "zip",
            "cooldown_seconds": args.sweep_cooldown,
            "iteration_order": args.iteration_order,
            "same_seed": args.same_seed,
            "parameters": {
                "phases.warmup.rate": list(rates),
                "phases.profiling.rate": list(rates),
                **(
                    {}
                    if args.requests_per_rate is not None
                    else {"phases.profiling.duration": list(durations)}
                ),
            },
        },
        "multi_run": {
            "num_runs": args.runs,
            "cooldown_seconds": args.run_cooldown,
            "confidence_level": args.confidence_level,
            "set_consistent_seed": True,
            "disable_warmup_after_first": args.warmup_once,
        },
    }


def read_aiperf_version() -> str | None:
    try:
        return importlib.metadata.version("aiperf")
    except importlib.metadata.PackageNotFoundError:
        pass

    executable = shutil.which("aiperf")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = f"{completed.stdout}\n{completed.stderr}"
    match = re.search(r"\b(\d+\.\d+(?:\.\d+)?(?:[-+._A-Za-z0-9]*)?)\b", text)
    return match.group(1) if match else None


def run_and_tee(command: list[str], *, cwd: Path, log_path: Path) -> None:
    print("\n$", shlex.join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {shlex.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="")
                log.write(line)
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def resolve_extra_inputs(args: argparse.Namespace) -> list[str]:
    """The AIPerf `--extra-inputs` this cell actually sends.

    `--no-default-extra-inputs` drops the campaign default of
    `ignore_eos:true temperature:0`; `--extra-input KEY:VALUE` adds to whatever
    survives. The manifest reports this list rather than restating the default,
    so a cell run without ignore_eos cannot be read later as a forced-length one.
    """
    extra_inputs: list[str] = []
    if not args.no_default_extra_inputs:
        extra_inputs.extend(["ignore_eos:true", "temperature:0"])
    extra_inputs.extend(args.extra_input)
    return extra_inputs


def build_profile_command(
    args: argparse.Namespace, yaml_path: Path, base_url: str
) -> list[str]:
    command = [
        "aiperf",
        "profile",
        "--config",
        str(yaml_path),
        "--tokenizer",
        args.tokenizer,
        "--ui",
        args.ui,
        "--export-level",
        args.export_level,
        "--slice-duration",
        str(args.slice_duration),
    ]

    if args.tokenizer_revision:
        command.extend(["--tokenizer-revision", args.tokenizer_revision])
    if args.tokenizer_trust_remote_code:
        command.append("--tokenizer-trust-remote-code")
    if args.apply_chat_template:
        command.append("--apply-chat-template")
    if args.use_server_token_count:
        command.append("--use-server-token-count")

    extra_inputs = resolve_extra_inputs(args)
    if extra_inputs:
        command.append("--extra-inputs")
        command.extend(extra_inputs)

    if args.no_server_metrics:
        command.append("--no-server-metrics")
    else:
        command.extend(["--server-metrics", base_url + "/metrics"])
    if args.no_gpu_telemetry:
        command.append("--no-gpu-telemetry")
    elif args.gpu_telemetry:
        command.append("--gpu-telemetry")
        command.extend(args.gpu_telemetry)

    for header in args.header:
        command.extend(["--header", header])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and run an AIPerf finite request-rate sweep for one fixed "
            "serving configuration."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    required = parser.add_argument_group("fixed deployment")
    required.add_argument("--config-id", required=True)
    required.add_argument("--model", required=True)
    required.add_argument("--tokenizer", help="HF tokenizer ID or local path; defaults to --model")
    required.add_argument("--tokenizer-revision")
    required.add_argument("--tokenizer-trust-remote-code", action="store_true")
    required.add_argument("--url", required=True)
    required.add_argument("--gpu-count", type=int, required=True)
    required.add_argument(
        "--server-container",
        help="Optional local Docker container to receipt and verify prefix caching against",
    )

    rate = parser.add_argument_group("offered-load sweep")
    exclusive = rate.add_mutually_exclusive_group(required=False)
    exclusive.add_argument("--rates", help="Required unless --until-knee or --capacity-output-tok-s")
    exclusive.add_argument("--capacity-output-tok-s", type=float)
    rate.add_argument("--rate-ratios", default=DEFAULT_RATE_RATIOS)

    ladder = parser.add_argument_group("knee-finding ladder (module docstring)")
    ladder.add_argument(
        "--until-knee",
        action="store_true",
        help=(
            "After the fixed --rates (which may be omitted), step +0.1 x "
            "--lambda-sat from --start-ratio until the knee is confirmed"
        ),
    )
    ladder.add_argument("--lambda-sat", type=float, help="Saturation rate in req/s the ratios scale")
    ladder.add_argument("--start-ratio", type=float, default=LADDER_START_RATIO)
    ladder.add_argument("--ladder-step", type=float, default=LADDER_STEP)
    ladder.add_argument("--knee-fraction", type=float, default=KNEE_ACHIEVED_FRACTION)
    ladder.add_argument("--plateau-tolerance", type=float, default=PLATEAU_TOLERANCE)
    ladder.add_argument("--confirmation-factor", type=float, default=CONFIRMATION_FACTOR)
    ladder.add_argument("--max-ladder-rungs", type=int, default=MAX_LADDER_RUNGS)
    ladder.add_argument(
        "--knee-achieved",
        choices=KNEE_ACHIEVED_BASES,
        default="tail-corrected",
        help="What 'achieved request rate' means in the knee test",
    )
    ladder.add_argument(
        "--tail-baseline-s",
        type=float,
        help=(
            "Drain tail (s) of a lighter rung measured earlier, for a ladder "
            "that starts with no fixed --rates; without it the lightest rung "
            "this invocation runs is the baseline, or the raw field is used"
        ),
    )
    ladder.add_argument(
        "--prior-output-tok-s",
        type=float,
        help="Output tok/s of the rung just below the ladder, seeding the gain test",
    )
    rate.add_argument(
        "--requests-per-rate",
        type=int,
        help=(
            "Use exactly this many sent requests in every rate cell instead of "
            "duration-based stopping; with --same-seed this holds prompt ordering fixed"
        ),
    )
    rate.add_argument("--target-requests", type=int, default=800)
    rate.add_argument("--min-duration", type=int, default=180)
    rate.add_argument("--max-duration", type=int, default=1200)
    rate.add_argument("--warmup-requests", type=int)
    rate.add_argument("--warmup-duration", type=int, default=60)
    rate.add_argument("--warmup-grace-period", type=int, default=300)
    rate.add_argument("--grace-period", type=int, default=900)
    rate.add_argument(
        "--max-concurrency",
        type=int,
        help="Emergency cap only; leave unset for a true open-loop sweep",
    )
    rate.add_argument("--arrival-pattern", choices=["poisson", "constant"], default="poisson")

    workload = parser.add_argument_group("AWS workload surrogate")
    workload.add_argument("--isl-p50", type=float, default=3500)
    workload.add_argument("--isl-p90", type=float, default=10000)
    workload.add_argument("--isl-cap", type=int, default=11200)
    workload.add_argument("--osl-p50", type=float, default=200)
    workload.add_argument("--osl-p90", type=float, default=400)
    workload.add_argument("--osl-cap", type=int, default=800)
    workload.add_argument(
        "--chat-template-overhead-tokens",
        type=int,
        default=0,
        help=(
            "Fixed chat-template tokens to subtract from generated ISL quantiles; "
            "AIPerf 0.12 does not compensate sequence_distribution entries"
        ),
    )
    workload.add_argument(
        "--workload-file",
        type=Path,
        help=(
            "Frozen AIPerf single_turn JSONL (workload/aws-p50p90-v1/requests-<tokenizer>.jsonl). "
            "When given, the driver replays this exact list sequentially instead of "
            "regenerating the bounded-lognormal sample, and the cell defaults to "
            "160 requests / 8 warmups / one run"
        ),
    )
    workload.add_argument(
        "--workload-id",
        help="Workload artifact id recorded in the manifest, e.g. aws-p50p90-v1",
    )
    workload.add_argument("--prompt-corpus", choices=["coding", "sonnet"], default="coding")
    workload.add_argument("--dataset-entries", type=int, default=4096)
    workload.add_argument("--seed", type=int, default=42)
    workload.add_argument("--apply-chat-template", action=argparse.BooleanOptionalAction, default=True)
    workload.add_argument("--use-server-token-count", action=argparse.BooleanOptionalAction, default=True)
    workload.add_argument("--extra-input", action="append", default=[], metavar="KEY:VALUE")
    workload.add_argument("--no-default-extra-inputs", action="store_true")
    workload.add_argument("--header", action="append", default=[], metavar="KEY:VALUE")

    rigor = parser.add_argument_group("repetitions and artifacts")
    rigor.add_argument("--runs", type=int)
    rigor.add_argument("--confidence-level", type=float, default=0.95)
    rigor.add_argument("--run-cooldown", type=float, default=30.0)
    rigor.add_argument("--sweep-cooldown", type=float, default=45.0)
    rigor.add_argument(
        "--iteration-order",
        choices=["repeated", "independent"],
        default="repeated",
    )
    rigor.add_argument("--same-seed", action=argparse.BooleanOptionalAction, default=True)
    rigor.add_argument("--warmup-once", action="store_true")
    rigor.add_argument("--slice-duration", type=float, default=30.0)
    rigor.add_argument("--endpoint-timeout", type=float, default=3600.0)
    rigor.add_argument("--artifact-root", type=Path, default=Path("./aws_rate_sweep_artifacts"))
    rigor.add_argument("--export-level", choices=["summary", "records", "raw"], default="records")
    rigor.add_argument("--ui", choices=["none", "simple"], default="simple")
    rigor.add_argument("--gpu-telemetry", action="append", default=[])
    rigor.add_argument("--no-gpu-telemetry", action="store_true")
    rigor.add_argument("--no-server-metrics", action="store_true")

    execution = parser.add_argument_group("execution")
    execution.add_argument("--generate-only", action="store_true")
    execution.add_argument("--skip-version-check", action="store_true")
    return parser.parse_args()


def rung_rate_value(rate: float) -> float:
    """The float a rung is named by: 6 significant digits, the campaign's key."""
    return float(f"{rate:.6g}")


def run_one_rung(
    args: argparse.Namespace,
    *,
    rate: float,
    artifact_root: Path,
    log_path: Path,
    base_url: str,
    chat_url: str,
) -> dict[str, Any]:
    """One AIPerf run at one offered rate, into rate_<x>__rate_<x>/.

    The directory name mirrors what AIPerf's own multi-rate sweep writes, so a
    ladder rung and a fixed rung look identical on disk. Returns the rung's
    summary numbers and timing.
    """
    value = rung_rate_value(rate)
    rung_dir = artifact_root / f"rate_{value}__rate_{value}"
    if rung_dir.exists() and any(rung_dir.iterdir()):
        raise SystemExit(f"Rung directory is not empty: {rung_dir}")
    rung_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = artifact_root / f"aiperf_rate_{value}.yaml"
    durations = compute_durations(
        [value],
        target_requests=args.target_requests,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
    config = build_config(args, [value], durations, rung_dir, chat_url)
    yaml_path.write_text(
        yaml.safe_dump(config, sort_keys=False, width=120), encoding="utf-8"
    )
    run_and_tee(
        ["aiperf", "config", "validate", "--config-file", str(yaml_path)],
        cwd=artifact_root,
        log_path=log_path,
    )
    run_and_tee(
        build_profile_command(args, yaml_path, base_url), cwd=artifact_root, log_path=log_path
    )
    exports = sorted(
        rung_dir.rglob("profile_export_aiperf.json"),
        key=lambda path: len(path.relative_to(rung_dir).parts),
    )
    if not exports:
        raise SystemExit(f"Rung at {value} req/s produced no profile_export_aiperf.json under {rung_dir}")
    export = exports[0]
    summary = json.loads(export.read_text(encoding="utf-8"))
    timing = rung_timing(profiling_records(export.parent))
    return {
        "rate": value,
        "export": str(export),
        "export_sha256": hashlib.sha256(export.read_bytes()).hexdigest(),
        "config_file": str(yaml_path),
        "request_count": int(export_metric(summary, "request_count") or 0),
        "errors": export_error_count(summary),
        "request_throughput_raw": export_metric(summary, "request_throughput"),
        "output_tok_s": export_metric(summary, "output_token_throughput") or 0.0,
        "benchmark_duration_s": export_metric(summary, "benchmark_duration"),
        **timing,
    }


def run_until_knee(
    args: argparse.Namespace,
    *,
    rates: list[float],
    artifact_root: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    log_path: Path,
    base_url: str,
    chat_url: str,
) -> int:
    """The fixed rungs, then the knee ladder; every rung its own AIPerf run."""
    results: list[dict[str, Any]] = []
    tail_baseline_s = args.tail_baseline_s
    basis = args.knee_achieved

    def save(status: str) -> None:
        manifest["status"] = status
        manifest["offered_rates_qps"] = [entry["rate"] for entry in results]
        manifest["rungs_run"] = len(results)
        manifest["rungs"] = results
        manifest["ladder"]["tail_baseline_s"] = tail_baseline_s
        manifest["ladder"]["knee_achieved_basis"] = basis
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def measure(rate: float, kind: str) -> dict[str, Any]:
        nonlocal tail_baseline_s, basis
        print(f"\n=== rung {kind}: {rate:.6g} req/s ({rate / args.lambda_sat:.3g} x lambda_sat)")
        entry = run_one_rung(
            args,
            rate=rate,
            artifact_root=artifact_root,
            log_path=log_path,
            base_url=base_url,
            chat_url=chat_url,
        )
        entry["kind"] = kind
        entry["ratio"] = entry["rate"] / args.lambda_sat
        if tail_baseline_s is None and args.knee_achieved == "tail-corrected":
            # The lightest rung this invocation runs: below the knee, its
            # drain tail is one request's unloaded latency.
            tail_baseline_s = entry["drain_tail_s"]
            entry["tail_baseline_source"] = "this rung (lightest of the run)"
        if args.knee_achieved == "tail-corrected" and tail_baseline_s is not None:
            entry["achieved_qps"] = tail_corrected_qps(
                request_count=entry["request_count"],
                duration_s=entry["duration_s"],
                tail_baseline_s=tail_baseline_s,
            )
            entry["achieved_basis"] = "tail-corrected"
        else:
            entry["achieved_qps"] = entry["request_throughput_raw"] or 0.0
            entry["achieved_basis"] = "raw"
            basis = "raw"
        entry["achieved_fraction"] = entry["achieved_qps"] / entry["rate"]
        results.append(entry)
        save("RUNNING")
        print(
            f"    achieved {entry['achieved_qps']:.4g} req/s ({entry['achieved_fraction']:.3f} of "
            f"offered, {entry['achieved_basis']}), {entry['output_tok_s']:.1f} output tok/s, "
            f"drain tail {entry['drain_tail_s']:.1f}s, errors {entry['errors']}"
        )
        return entry

    save("RUNNING")
    try:
        prior_tok_s = args.prior_output_tok_s
        for rate in rates:
            entry = measure(rate, "fixed")
            if entry["errors"]:
                manifest["stop_reason"] = "errors"
                manifest["knee_rate"] = None
                manifest["knee_ratio"] = None
                save("FAILED")
                raise SystemExit(f"Fixed rung {rate:.6g} req/s had {entry['errors']} errored requests")
            prior_tok_s = entry["output_tok_s"]

        def run_rung(rate: float) -> tuple[float, float, int]:
            entry = measure(rate, "ladder")
            return entry["achieved_qps"], entry["output_tok_s"], entry["errors"]

        verdict = knee_ladder(
            run_rung,
            lambda_sat=args.lambda_sat,
            start_ratio=args.start_ratio,
            step=args.ladder_step,
            knee_fraction=args.knee_fraction,
            plateau_tolerance=args.plateau_tolerance,
            confirmation_factor=args.confirmation_factor,
            prior_output_tok_s=prior_tok_s,
            max_rungs=args.max_ladder_rungs,
        )
    except BaseException:
        if manifest.get("status") != "FAILED":
            save("FAILED")
        raise

    # Label the ladder's rungs with the pure function's verdict (same order).
    ladder_entries = [entry for entry in results if entry["kind"] == "ladder"]
    for entry, rung in zip(ladder_entries, verdict["rungs"], strict=True):
        entry["kind"] = "confirmation" if rung["kind"] == "confirmation" else "ladder"
        entry["knee"] = rung["knee"]
        for key in ("knee_cleared", "confirms_ratio", "gain_vs_previous", "gain_vs_knee"):
            if key in rung:
                entry[key] = rung[key]
    manifest["knee_rate"] = verdict["knee_rate"]
    manifest["knee_ratio"] = verdict["knee_ratio"]
    manifest["knee_confirmed"] = verdict["knee_confirmed"]
    manifest["stop_reason"] = verdict["stop_reason"]
    manifest["stop_ratio"] = verdict["stop_ratio"]
    manifest["ladder"]["verdict"] = verdict
    manifest["aggregate_file"] = None
    manifest["aggregate_basis"] = (
        "until-knee: one AIPerf run per rung, no AIPerf sweep aggregate; "
        "read rungs[] here or rate_<x>__rate_<x>/profile_export_aiperf.json"
    )
    save("COMPLETE")
    print(
        f"\nLadder finished: {verdict['rungs_run']} ladder rungs, stop_reason "
        f"{verdict['stop_reason']}, knee_rate {verdict['knee_rate']}, "
        f"knee_ratio {verdict['knee_ratio']}"
    )
    return 0


def main() -> int:
    args = parse_args()
    args.tokenizer = args.tokenizer or args.model

    if args.workload_file is not None:
        args.workload_file = args.workload_file.resolve()
        if not args.workload_file.is_file():
            raise SystemExit(f"--workload-file does not exist: {args.workload_file}")
        workload_records = sum(
            1
            for line in args.workload_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if workload_records < 1:
            raise SystemExit(f"--workload-file is empty: {args.workload_file}")
        if args.requests_per_rate is None:
            args.requests_per_rate = 160
        if args.warmup_requests is None:
            args.warmup_requests = 8
        if args.runs is None:
            args.runs = 1
    else:
        workload_records = None
    if args.runs is None:
        args.runs = 3

    if args.gpu_count < 1:
        raise SystemExit("--gpu-count must be >= 1")
    if not 1 <= args.runs <= 10:
        raise SystemExit("--runs must be between 1 and 10")
    if not 0 < args.confidence_level < 1:
        raise SystemExit("--confidence-level must be between 0 and 1")
    if args.max_concurrency is not None and args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be >= 1")
    if args.requests_per_rate is not None and args.requests_per_rate < 1:
        raise SystemExit("--requests-per-rate must be >= 1")
    if args.warmup_requests is not None and args.warmup_requests < 1:
        raise SystemExit("--warmup-requests must be >= 1")
    if not 0 <= args.chat_template_overhead_tokens < args.isl_p50:
        raise SystemExit(
            "--chat-template-overhead-tokens must be non-negative and below ISL p50"
        )

    isl = bounded_lognormal(p50=args.isl_p50, p90=args.isl_p90, cap=args.isl_cap)
    osl = bounded_lognormal(p50=args.osl_p50, p90=args.osl_p90, cap=args.osl_cap)
    maximum_total_tokens = args.isl_cap + args.osl_cap
    base_url, chat_url = normalize_chat_url(args.url)
    endpoint_receipt = None
    container_receipt = None
    if not args.generate_only:
        endpoint_receipt = validate_endpoint(base_url, args.model, maximum_total_tokens)
        container_receipt = collect_container_receipt(args.server_container)

    if args.tokenizer_revision is None and not Path(args.tokenizer).exists():
        print(
            "WARNING: tokenizer revision is not pinned. Pass --tokenizer-revision "
            "for final reproducible AWS measurements.",
            file=sys.stderr,
        )

    rates, estimated_saturation_qps = compute_rates(args)
    durations = compute_durations(
        rates,
        target_requests=args.target_requests,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )

    artifact_root = (args.artifact_root / args.config_id).resolve()
    if artifact_root.exists() and any(artifact_root.iterdir()):
        raise SystemExit(f"Artifact directory is not empty: {artifact_root}")
    artifact_root.mkdir(parents=True, exist_ok=True)
    yaml_path = artifact_root / "aiperf_rate_sweep.yaml"
    manifest_path = artifact_root / "aws_sweep_manifest.json"
    log_path = artifact_root / "driver.log"

    if args.until_knee:
        if args.lambda_sat is None or args.lambda_sat <= 0:
            raise SystemExit("--until-knee requires --lambda-sat > 0")
        if args.requests_per_rate is None:
            raise SystemExit(
                "--until-knee needs --requests-per-rate: the ladder compares "
                "rungs of an identical request count"
            )
        # One AIPerf run per rung, each into its own rate_<x>__rate_<x>/; the
        # sweep YAML below is not used.
        profile_command = None
    else:
        config = build_config(args, rates, durations, artifact_root, chat_url)
        yaml_path.write_text(
            yaml.safe_dump(config, sort_keys=False, width=120), encoding="utf-8"
        )
        profile_command = build_profile_command(args, yaml_path, base_url)
    resolved_extra_inputs = resolve_extra_inputs(args)
    if args.workload_file is not None:
        check_extra_inputs(args.workload_file, resolved_extra_inputs)
    resolved_extra_map = dict(
        part.split(":", 1) for part in resolved_extra_inputs if ":" in part
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "status": "GENERATED" if args.generate_only else "READY",
        "config_id": args.config_id,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "url": chat_url,
        "gpu_count": args.gpu_count,
        "aiperf_version_required": AIPERF_VERSION,
        "endpoint_receipt": endpoint_receipt,
        "container_receipt": container_receipt,
        "workload": {
            "isl": isl["receipt"],
            "osl": osl["receipt"],
            "maximum_total_tokens": maximum_total_tokens,
            "prompt_corpus": args.prompt_corpus,
            "extra_inputs": resolved_extra_inputs,
            "ignore_eos": resolved_extra_map.get("ignore_eos") == "true",
            "temperature": (
                float(resolved_extra_map["temperature"])
                if "temperature" in resolved_extra_map
                else None
            ),
            "apply_chat_template": args.apply_chat_template,
            "use_server_token_count": args.use_server_token_count,
            "workload_id": args.workload_id,
            "workload_file": (
                None if args.workload_file is None else str(args.workload_file)
            ),
            "workload_records": workload_records,
            "empirical_quantile_points": None if args.workload_file else 100,
            "chat_template_overhead_tokens": args.chat_template_overhead_tokens,
            "aiperf_representation": (
                (
                    "frozen single_turn JSONL replayed sequentially; per-request "
                    "output_length becomes max_tokens, and with ignore_eos the "
                    "realized OSL is exact - without it (see extra_inputs) the "
                    "request stops at EOS and realized OSL is shorter than "
                    "nominal. A cell of N requests over an N-record "
                    "list is one full cycle, so every cell and every deployment "
                    "sees the identical multiset; the 8 warmup requests rotate "
                    "the start index deterministically."
                )
                if args.workload_file is not None
                else (
                    "100 equal-weight fixed (ISL, OSL) pairs under prompts.sequence_distribution; "
                    "OSL quantiles use a seeded permutation to approximate independent marginals."
                )
            ),
            "assumption": (
                "Independent bounded-lognormal marginals; AWS supplied p50/p90, "
                "while the family and caps are an explicit surrogate assumption."
            ),
        },
        "offered_rates_qps": rates,
        "profiling_stop_condition": (
            {"requests_per_rate": args.requests_per_rate}
            if args.requests_per_rate is not None
            else {"durations_seconds": durations}
        ),
        "profiling_durations_seconds": (
            None if args.requests_per_rate is not None else durations
        ),
        "estimated_dispatch_durations_seconds": (
            [args.requests_per_rate / rate for rate in rates]
            if args.requests_per_rate is not None
            else durations
        ),
        "warmup_stop_condition": (
            {"requests": args.warmup_requests}
            if args.warmup_requests is not None
            else {"duration_seconds": args.warmup_duration}
        ),
        "estimated_saturation_qps": estimated_saturation_qps,
        "runs_per_rate": args.runs,
        "iteration_order": args.iteration_order,
        "same_seed_across_rates": args.same_seed,
        "true_open_loop": args.max_concurrency is None,
        "max_concurrency": args.max_concurrency,
        "profile_command": profile_command,
        "config_file": None if args.until_knee else str(yaml_path),
    }
    if args.until_knee:
        manifest["ladder"] = {
            "mode": "until-knee",
            "lambda_sat": args.lambda_sat,
            "fixed_rates_qps": rates,
            "start_ratio": args.start_ratio,
            "step": args.ladder_step,
            "knee_fraction": args.knee_fraction,
            "plateau_tolerance": args.plateau_tolerance,
            "confirmation_factor": args.confirmation_factor,
            "max_rungs": args.max_ladder_rungs,
            "knee_achieved": args.knee_achieved,
            "tail_baseline_s": args.tail_baseline_s,
            "prior_output_tok_s": args.prior_output_tok_s,
        }
        manifest["estimated_saturation_qps"] = args.lambda_sat
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.until_knee:
        manifest["config_sha256"] = hashlib.sha256(yaml_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote: {yaml_path}")
    print(f"Wrote: {manifest_path}")
    print("\nRate plan:")
    if args.requests_per_rate is not None:
        print("  offered QPS    requests    estimated dispatch(s)")
        for rate in rates:
            print(
                f"  {rate:11.5g}    {args.requests_per_rate:8d}    "
                f"{args.requests_per_rate / rate:21.1f}"
            )
    else:
        print("  offered QPS    duration(s)    expected arrivals")
        for rate, duration in zip(rates, durations, strict=True):
            print(f"  {rate:11.5g}    {duration:11d}    {rate * duration:17.1f}")
    if estimated_saturation_qps is not None:
        print(f"\nEstimated saturation QPS: {estimated_saturation_qps:.5g}")
    if args.max_concurrency is not None:
        print(
            "\nWARNING: --max-concurrency is present. If it binds, this is not a "
            "pure open-loop overload measurement.",
            file=sys.stderr,
        )

    if args.generate_only:
        print("\nGeneration complete; AIPerf was not invoked.")
        return 0

    if shutil.which("aiperf") is None:
        raise SystemExit("The 'aiperf' executable is not on PATH. Run: uv sync")

    version = read_aiperf_version()
    manifest["aiperf_version"] = version
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.skip_version_check and version != AIPERF_VERSION:
        raise SystemExit(
            f"This kit requires AIPerf {AIPERF_VERSION}, but found {version!r}."
        )

    if args.until_knee:
        return run_until_knee(
            args,
            rates=rates,
            artifact_root=artifact_root,
            manifest=manifest,
            manifest_path=manifest_path,
            log_path=log_path,
            base_url=base_url,
            chat_url=chat_url,
        )

    run_and_tee(
        ["aiperf", "config", "validate", "--config-file", str(yaml_path)],
        cwd=artifact_root,
        log_path=log_path,
    )
    expanded_path = artifact_root / "aiperf_rate_sweep_expanded.yaml"
    expanded = subprocess.run(
        [
            "aiperf",
            "config",
            "expand",
            "--config-file",
            str(yaml_path),
            "--full",
            "--format",
            "yaml",
        ],
        cwd=str(artifact_root),
        check=True,
        capture_output=True,
        text=True,
    )
    expanded_path.write_text(expanded.stdout, encoding="utf-8")
    print(f"Wrote: {expanded_path}")

    manifest["status"] = "RUNNING"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        run_and_tee(profile_command, cwd=artifact_root, log_path=log_path)
    except BaseException:
        manifest["status"] = "FAILED"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        raise
    def shallowest(name: str) -> list[Path]:
        # AIPerf also writes per-phase copies under phases/; the run-level export
        # is the shallowest match.
        matches = sorted(
            artifact_root.rglob(name),
            key=lambda path: len(path.relative_to(artifact_root).parts),
        )
        return matches[:1]

    aggregates = shallowest("profile_export_aiperf_sweep.json")
    if not aggregates and len(rates) == 1:
        # A single-rate, single-run cell has nothing to aggregate across, so
        # AIPerf writes no sweep file; the cell export is the result. Assert it
        # exists rather than reporting a cell that produced nothing.
        aggregates = shallowest("profile_export_aiperf.json")
    if len(aggregates) != 1:
        manifest["status"] = "FAILED_AGGREGATE_CHECK"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(
            f"Expected one result export below {artifact_root}, found {len(aggregates)}"
        )
    manifest["aggregate_file"] = str(aggregates[0])
    manifest["aggregate_sha256"] = hashlib.sha256(aggregates[0].read_bytes()).hexdigest()
    manifest["status"] = "COMPLETE"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("\nSweep finished.")
    print(
        "Plot with:\n  "
        f"python plot_aws_frontiers.py --root {shlex.quote(str(artifact_root))} "
        f"--output-dir {shlex.quote(str(artifact_root / 'plots'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
