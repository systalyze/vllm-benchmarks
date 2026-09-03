#!/usr/bin/env bash
# vllm branch: systalyze/dspark-gemma4-0.28.0
# Config 3: Nemotron-3-Nano-30B-A3B on G7 (g7.48xlarge, 8x RTX PRO 4500 32 GB) (NVFP4; DP8×TP1; BF16 KV cache; max-num-seqs 16; max-num-batched-tokens 8192)
# model /data/models/Nemotron-3-Nano-30B-A3B-NVFP4
# image vllm/vllm-openai:v0.28.0
# GPUs [5] (TP1: 1 replica(s))
set -euo pipefail
BASE_PORT=${BASE_PORT:-8300}
# repository root, for the bind mounts (patched vLLM files from vllm/, the draft under models/)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

docker run -d --name aws-g7-nem-nvfp4-mns64-mnbt8192-v0280-gpu5 --gpus '"device=5"' --ipc=host --shm-size=16g -p $((BASE_PORT+0)):$((BASE_PORT+0)) -v /data:/data vllm/vllm-openai:v0.28.0 --model /data/models/Nemotron-3-Nano-30B-A3B-NVFP4 --served-model-name g7-nem-nvfp4-mns64-mnbt8192-v0280 --tensor-parallel-size 1 --port $((BASE_PORT+0)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --max-num-seqs 16 --max-num-batched-tokens 8192

# Stop ONLY these containers, by name:
#   docker stop aws-g7-nem-nvfp4-mns64-mnbt8192-v0280-gpu5
