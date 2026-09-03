#!/usr/bin/env python3
"""Which exact-length mechanism a frozen workload file uses, and the extra inputs it forbids.

A frozen single_turn JSONL holds its sampled OSL exactly in one of two ways:

- ignore_eos: plain records; the driver passes `--extra-inputs ignore_eos:true`.
- min_tokens: every record carries `"extra": {"min_tokens": <output_length>}`,
  so vLLM suppresses EOS until min_tokens and stops at max_tokens, the same
  number. Speculator rows use this one: ignore_eos would make the model keep
  generating after a natural EOS, which degenerates into repetition and
  inflates draft acceptance.

The two must not be mixed. With `ignore_eos:true` on a min_tokens file the stop
token is ignored anyway and min_tokens is redundant, so the run is a forced-length
run mislabelled as a natural one. A global `min_tokens:N` extra input on a
min_tokens file is silently overridden per request by the file's own value
(AIPerf merges `extra` after `--extra-inputs`), so the manifest would state a
number no request was sent with. `check_extra_inputs` refuses both.

Hook for a driver (one line, after it has resolved its extra inputs):

    check_extra_inputs(workload_file, extra_inputs)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXACT_LENGTH_MODES = ("ignore_eos", "min_tokens")


def exact_length_mode(workload_file: Path) -> str:
    """Read the mode off the file itself: every line carries extra.min_tokens, or none does."""
    with_min_tokens = 0
    total = 0
    with Path(workload_file).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            record = json.loads(line)
            extra = record.get("extra") or {}
            if "min_tokens" in extra:
                if extra["min_tokens"] != record.get("output_length"):
                    raise SystemExit(
                        f"{workload_file}: line {total - 1} has min_tokens="
                        f"{extra['min_tokens']!r} but output_length="
                        f"{record.get('output_length')!r}; the exact-length "
                        "variant pins them equal"
                    )
                with_min_tokens += 1
    if total == 0:
        raise SystemExit(f"{workload_file}: empty workload file")
    if with_min_tokens == 0:
        return "ignore_eos"
    if with_min_tokens == total:
        return "min_tokens"
    raise SystemExit(
        f"{workload_file}: {with_min_tokens} of {total} lines carry extra.min_tokens; "
        "a workload file is one mode or the other"
    )


def check_extra_inputs(workload_file: Path, extra_inputs: list[str]) -> str:
    """Refuse extra inputs that contradict the file's mode; return the mode."""
    mode = exact_length_mode(workload_file)
    keys = {part.split(":", 1)[0]: part.split(":", 1)[1] for part in extra_inputs if ":" in part}
    if mode == "min_tokens":
        if keys.get("ignore_eos", "false").lower() == "true":
            raise SystemExit(
                f"{workload_file} pins the length with per-request min_tokens; "
                "--extra-inputs ignore_eos:true must not be passed with it "
                "(it would run forced-length generation past EOS while the "
                "workload claims natural stopping)"
            )
        if "min_tokens" in keys:
            raise SystemExit(
                f"{workload_file} carries per-request min_tokens; a global "
                f"min_tokens:{keys['min_tokens']} extra input is overridden per "
                "request by AIPerf and would mislabel the manifest"
            )
    return mode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload_file", type=Path)
    parser.add_argument("--extra-input", action="append", default=[], metavar="KEY:VALUE")
    args = parser.parse_args()
    print(check_extra_inputs(args.workload_file, args.extra_input))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
