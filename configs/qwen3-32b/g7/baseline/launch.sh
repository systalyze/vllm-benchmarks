#!/usr/bin/env bash
# vllm branch: systalyze/serving-0.27.1
# Baseline: Qwen3-32B on G7 (g7.48xlarge, 8x RTX PRO 4500 32 GB), stock vLLM, BF16 checkpoint
# model /data/models/Qwen3-32B
# image vllm/vllm-openai:v0.27.1
# GPUs [0, 1, 2, 3, 4, 5, 6, 7] (TP4: 2 replica(s))
set -euo pipefail
BASE_PORT=${BASE_PORT:-8300}
# repository root, for the bind mounts (patched vLLM files from vllm/, the draft under models/)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

docker run -d --name aws-g7-base-qwen-bf16-tp4-gpu0 --gpus '"device=0,1,2,3"' --ipc=host --shm-size=16g -p $((BASE_PORT+0)):$((BASE_PORT+0)) -v /data:/data vllm/vllm-openai:v0.27.1 --model /data/models/Qwen3-32B --served-model-name g7-base-qwen-bf16-tp4 --tensor-parallel-size 4 --port $((BASE_PORT+0)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching
docker run -d --name aws-g7-base-qwen-bf16-tp4-gpu4 --gpus '"device=4,5,6,7"' --ipc=host --shm-size=16g -p $((BASE_PORT+1)):$((BASE_PORT+1)) -v /data:/data vllm/vllm-openai:v0.27.1 --model /data/models/Qwen3-32B --served-model-name g7-base-qwen-bf16-tp4 --tensor-parallel-size 4 --port $((BASE_PORT+1)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching

# Stop ONLY these containers, by name:
#   docker stop aws-g7-base-qwen-bf16-tp4-gpu0
#   docker stop aws-g7-base-qwen-bf16-tp4-gpu4
