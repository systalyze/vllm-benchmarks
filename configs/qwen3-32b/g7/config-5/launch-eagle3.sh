#!/usr/bin/env bash
# vllm branch: systalyze/dspark-gemma4-0.28.0
# Config 5 + EAGLE-3 speculation: Qwen3-32B on G7 (g7.48xlarge, 8x RTX PRO 4500 32 GB) (NVFP4; DP4×TP2; FP8 KV cache; max-num-seqs 8; max-num-batched-tokens 2048)
# model /data/models/Qwen3-32B-NVFP4
# image vllm/vllm-openai:v0.28.0
# GPUs [6, 7] (TP2: 1 replica(s))
# speculative decoding: eagle3, 3 draft tokens, draft /data/models/Qwen3-32B-speculator.eagle3
set -euo pipefail
BASE_PORT=${BASE_PORT:-8300}
# repository root, for the bind mounts (patched vLLM files from vllm/, the draft under models/)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

docker run -d --name aws-spec2-g7-qwen-opt5-eagle3-k3-gpu6 --gpus '"device=6,7"' --ipc=host --shm-size=16g -p $((BASE_PORT+0)):$((BASE_PORT+0)) -v /data:/data vllm/vllm-openai:v0.28.0 --model /data/models/Qwen3-32B-NVFP4 --served-model-name spec2-g7-qwen-opt5-eagle3-k3 --tensor-parallel-size 2 --port $((BASE_PORT+0)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --max-num-seqs 8 --max-num-batched-tokens 2048 --kv-cache-dtype fp8 --gpu-memory-utilization 0.80 --speculative-config '{"method": "eagle3", "model": "/data/models/Qwen3-32B-speculator.eagle3", "num_speculative_tokens": 3}'

# Stop ONLY these containers, by name:
#   docker stop aws-spec2-g7-qwen-opt5-eagle3-k3-gpu6
