#!/usr/bin/env bash
# bench/run.sh <config folder> [--arm dspark|eagle3] [--dry-run] [run.py options...]
#
# Reproduces one configuration from configs/<model>/<instance>/<config-N|baseline>/:
#   1. picks its launch.sh + row.json (or launch-<arm>.sh + row-<arm>.json with --arm),
#   2. checks out, in the vllm/ submodule, the branch named by the
#      "# vllm branch: ..." line at the top of that launch.sh (the launch line
#      bind-mounts patched files from the submodule over the stock image),
#   3. hands the deployment to bench/run.py, which boots it, probes saturation,
#      runs the fixed rungs and the plateau ladder, and tears it down
#      (512-request rungs for max-num-seqs >= 128, the exact-length workload
#      for a speculator arm, exactly as the campaign did).
# --dry-run prints every command that would run and executes none of them.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
DEFAULT_BRANCH=systalyze/serving-0.27.1

if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  sed -n '2,14p' "$0"; exit 1
fi
CONFIG="$1"; shift
ARM=""; DRY=0; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --arm) ARM="$2"; shift 2 ;;
    --arm=*) ARM="${1#--arm=}"; shift ;;
    --dry-run) DRY=1; ARGS+=("$1"); shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
CONFIG="$(cd "$CONFIG" && pwd)"
if [ -n "$ARM" ]; then
  LAUNCH="$CONFIG/launch-$ARM.sh"; ROW="$CONFIG/row-$ARM.json"
else
  LAUNCH="$CONFIG/launch.sh"; ROW="$CONFIG/row.json"
fi
for f in "$LAUNCH" "$ROW"; do
  [ -f "$f" ] || { echo "missing: $f (see configs/INDEX.md)" >&2; exit 1; }
done

BRANCH="$(sed -n 's/^# vllm branch: *//p' "$LAUNCH" | head -1)"
BRANCH="${BRANCH:-$DEFAULT_BRANCH}"
[ -f "$REPO/vllm/setup.py" ] || { echo "vllm/ submodule is empty: git submodule update --init" >&2; exit 1; }
echo "# vllm/ submodule: git -C $REPO/vllm checkout $BRANCH"
if [ "$DRY" = 1 ]; then
  echo "# --dry-run: checkout skipped (currently on $(git -C "$REPO/vllm" rev-parse --abbrev-ref HEAD))"
else
  git -C "$REPO/vllm" checkout -q "$BRANCH"
  echo "# vllm/ is at $(git -C "$REPO/vllm" rev-parse --short HEAD) ($BRANCH)"
fi

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$HERE/.venv/bin/python" ]; then PY="$HERE/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
exec "$PY" "$HERE/run.py" "$ROW" --launch "$LAUNCH" "${ARGS[@]}"
