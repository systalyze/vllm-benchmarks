#!/usr/bin/env bash
# vllm branch: systalyze/serving-0.27.1
# Baseline: Gemma-4-26B-A4B-IT on G7e (g7e.24xlarge, 4x RTX PRO 6000 96 GB), stock vLLM, BF16 checkpoint
# model /data/models/gemma-4-26B-A4B-it
# image systalyze/vllm-openai:v0.27.1-tf5.14.1
# GPUs [0] (TP1: 1 replica(s))
set -euo pipefail
BASE_PORT=${BASE_PORT:-8300}
# repository root, for the bind mounts (patched vLLM files from vllm/, the draft under models/)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

docker run -d --name aws-g7e-gem-ctl-r512-gpu0 --gpus '"device=0"' --ipc=host --shm-size=16g -p $((BASE_PORT+0)):$((BASE_PORT+0)) -v /data:/data systalyze/vllm-openai:v0.27.1-tf5.14.1 --model /data/models/gemma-4-26B-A4B-it --served-model-name g7e-gem-ctl-r512 --tensor-parallel-size 1 --port $((BASE_PORT+0)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching

# Stop ONLY these containers, by name:
#   docker stop aws-g7e-gem-ctl-r512-gpu0
