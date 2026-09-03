#!/usr/bin/env python3
"""Build the frozen AWS p50/p90 workload: one AIPerf single_turn JSONL per tokenizer.

ISL and OSL are inverse-CDF draws of the bounded lognormals declared in
configs/aiperf-baseline.json at evenly spaced quantiles (i+0.5)/N, so the
realized p50/p90 hit the AWS targets by construction rather than by sampling
luck. The (ISL, OSL) pairs are identical for every tokenizer and every
deployment; only the prompt text differs, because ISL is measured in the
served model's own tokens.

Prompt content is AIPerf's own `coding` corpus generator, the same corpus the
earlier synthetic sweeps used.

Output per tokenizer: requests-<tokenizer-id>.jsonl, whose lines AIPerf 0.12.0
consumes as dataset.type=file / format=single_turn / sampling=sequential.

--exact-length-mode picks how a request is held to its sampled OSL:

- ignore_eos (the curve rows): the record is `text` + `output_length` +
  `session_id`; the driver adds `--extra-inputs ignore_eos:true`, so vLLM keeps
  generating past EOS until max_tokens = output_length.
- min_tokens (the speculator rows): every record additionally carries
  `"extra": {"min_tokens": <output_length>}`, which AIPerf shallow-merges into
  the request body, so vLLM suppresses EOS until min_tokens and stops at
  max_tokens = the same number, WITHOUT ignore_eos. Speculator rows cannot use
  ignore_eos: text generated after a natural EOS degenerates into repetition
  and inflates draft acceptance. Prompts, lengths and order are identical in
  both modes; only the `extra` field differs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import NormalDist

from transformers import AutoTokenizer

from aiperf.common import random_generator as rng
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.dataset.content import PromptConfig
from aiperf.dataset.generator.coding_content import CodingContentGenerator

EXACT_LENGTH_MODES = ("ignore_eos", "min_tokens")


def quantile_targets(*, p50: float, p90: float, cap: int, count: int) -> list[int]:
    """Inverse-CDF of the bounded lognormal at quantiles (i+0.5)/count."""
    sigma = math.log(p90 / p50) / NormalDist().inv_cdf(0.9)
    values = []
    for index in range(count):
        probability = (index + 0.5) / count
        value = math.exp(math.log(p50) + sigma * NormalDist().inv_cdf(probability))
        values.append(max(1, min(cap, round(value))))
    # Pin the two order statistics that linear-interpolated p50 and p90 read, so
    # the realized percentiles equal the AWS targets exactly rather than to
    # within the quantile grid's discretization.
    for probability, target in ((0.50, p50), (0.90, p90)):
        position = probability * (count - 1)
        low = math.floor(position)
        high = math.ceil(position)
        values[low] = values[high] = min(cap, round(target))
    if values != sorted(values):
        raise SystemExit("pinning p50/p90 broke the monotone quantile ladder")
    return values


def template_length(tok, text: str) -> int:
    """Token count of one user turn after the chat template, as the server counts it."""
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return len(tok.encode(rendered, add_special_tokens=False))


def build_for_tokenizer(
    *,
    tokenizer_id: str,
    tokenizer_path: Path,
    isl_targets: list[int],
    osl_targets: list[int],
    seed: int,
    max_passes: int,
    exact_length_mode: str,
) -> tuple[list[dict], dict]:
    hf = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=False)
    rng.reset()
    rng.init(seed)
    generator = CodingContentGenerator(
        config=PromptConfig(), tokenizer=Tokenizer.from_pretrained(str(tokenizer_path))
    )

    overhead = template_length(hf, "")
    records: list[dict] = []
    realized: list[int] = []
    passes_used = 0
    for index, (isl, osl) in enumerate(zip(isl_targets, osl_targets, strict=True)):
        budget = max(1, isl - overhead)
        text = generator.generate_prompt(budget)
        length = template_length(hf, text)
        for _ in range(max_passes):
            if length == isl:
                break
            passes_used += 1
            budget = max(1, budget + (isl - length))
            text = generator.generate_prompt(budget)
            length = template_length(hf, text)
        realized.append(length)
        record = {"text": text, "output_length": osl, "session_id": f"aws-p50p90-{index:03d}"}
        if exact_length_mode == "min_tokens":
            # Per-request extra body: AIPerf's single_turn loader maps `extra` to
            # Turn.extra_body, which the chat formatter merges into the payload
            # after --extra-inputs, so this pins min_tokens = max_tokens.
            record["extra"] = {"min_tokens": osl}
        records.append(record)
    return records, {
        "chat_template_overhead_tokens": overhead,
        "correction_passes": passes_used,
        "realized_isl": realized,
    }


def percentiles(values: list[int]) -> dict[str, float]:
    ordered = sorted(values)

    def q(p: float) -> float:
        position = p * (len(ordered) - 1)
        low = math.floor(position)
        high = math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "min": ordered[0],
        "p50": q(0.50),
        "p90": q(0.90),
        "p99": q(0.99),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "sum": sum(ordered),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--isl-p50", type=float, default=3500)
    parser.add_argument("--isl-p90", type=float, default=10000)
    parser.add_argument("--isl-cap", type=int, default=11200)
    parser.add_argument("--osl-p50", type=float, default=200)
    parser.add_argument("--osl-p90", type=float, default=400)
    parser.add_argument("--osl-cap", type=int, default=800)
    parser.add_argument("--joint-cap", type=int, default=12000)
    parser.add_argument("--max-passes", type=int, default=6)
    parser.add_argument(
        "--exact-length-mode",
        choices=EXACT_LENGTH_MODES,
        default="ignore_eos",
        help=(
            "ignore_eos: plain records, the driver forces the length with "
            "--extra-inputs ignore_eos:true. min_tokens: every record carries "
            "extra.min_tokens = output_length so the length is exact without "
            "ignore_eos (speculator rows; ignore_eos must NOT be passed)"
        ),
    )
    parser.add_argument(
        "--tokenizer",
        action="append",
        required=True,
        metavar="ID=SUBDIR",
        help="Repeatable: workload tokenizer id and its directory under --tokenizer-root",
    )
    args = parser.parse_args()

    isl_sorted = quantile_targets(
        p50=args.isl_p50, p90=args.isl_p90, cap=args.isl_cap, count=args.requests
    )
    osl_sorted = quantile_targets(
        p50=args.osl_p50, p90=args.osl_p90, cap=args.osl_cap, count=args.requests
    )
    # Independent marginals: decorrelate OSL from ISL, then shuffle the pair order.
    osl_shuffled = list(osl_sorted)
    random.Random(args.seed + 1).shuffle(osl_shuffled)
    pairs = list(zip(isl_sorted, osl_shuffled, strict=True))
    random.Random(args.seed).shuffle(pairs)
    over = [(isl, osl) for isl, osl in pairs if isl + osl > args.joint_cap]
    if over:
        raise SystemExit(f"{len(over)} pairs exceed the joint cap {args.joint_cap}: {over[:5]}")
    isl_targets = [isl for isl, _ in pairs]
    osl_targets = [osl for _, osl in pairs]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "workload_id": args.out_dir.name,
        "requests": args.requests,
        "seed": args.seed,
        "prompt_corpus": "coding",
        "construction": (
            "inverse-CDF of the bounded lognormal at quantiles (i+0.5)/N; OSL "
            "decorrelated with seed+1, pair order shuffled with the seed"
        ),
        "isl": {"p50": args.isl_p50, "p90": args.isl_p90, "cap": args.isl_cap},
        "osl": {"p50": args.osl_p50, "p90": args.osl_p90, "cap": args.osl_cap},
        "maximum_total_tokens": args.joint_cap,
        "exact_length_mode": args.exact_length_mode,
        "isl_targets": isl_targets,
        "osl_targets": osl_targets,
        "osl_realized_percentiles": percentiles(osl_targets),
        "aiperf": {
            "version": "0.12.0",
            "dataset_type": "file",
            "format": "single_turn",
            "sampling": "sequential",
            "note": (
                "output_length becomes the per-request max_tokens; combine with "
                "--extra-inputs ignore_eos:true so the realized OSL is exact"
                if args.exact_length_mode == "ignore_eos"
                else "output_length becomes the per-request max_tokens and "
                "extra.min_tokens pins the same number as min_tokens, so the "
                "realized OSL is exact WITHOUT ignore_eos; do not pass "
                "--extra-inputs ignore_eos:true with this file"
            ),
        },
        "tokenizers": {},
    }

    for spec in args.tokenizer:
        tokenizer_id, _, subdir = spec.partition("=")
        if not subdir:
            raise SystemExit(f"--tokenizer needs ID=SUBDIR, got {spec!r}")
        path = args.tokenizer_root / subdir
        records, receipt = build_for_tokenizer(
            tokenizer_id=tokenizer_id,
            tokenizer_path=path,
            isl_targets=isl_targets,
            osl_targets=osl_targets,
            seed=args.seed,
            max_passes=args.max_passes,
            exact_length_mode=args.exact_length_mode,
        )
        out = args.out_dir / f"requests-{tokenizer_id}.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        realized = receipt.pop("realized_isl")
        exact = sum(1 for a, b in zip(realized, isl_targets, strict=True) if a == b)
        summary["tokenizers"][tokenizer_id] = {
            "tokenizer_dir": str(path),
            "file": out.name,
            "chat_template_overhead_tokens": receipt["chat_template_overhead_tokens"],
            "correction_passes": receipt["correction_passes"],
            "requests_hitting_target_isl_exactly": exact,
            "realized_isl_percentiles": percentiles(realized),
            "realized_isl": realized,
        }
        print(
            f"{tokenizer_id}: overhead={receipt['chat_template_overhead_tokens']} "
            f"exact={exact}/{args.requests} "
            f"ISL p50={percentiles(realized)['p50']:.0f} p90={percentiles(realized)['p90']:.0f}"
        )

    (args.out_dir / "workload.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote: {args.out_dir / 'workload.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
