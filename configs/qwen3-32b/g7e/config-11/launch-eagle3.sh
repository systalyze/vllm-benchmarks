#!/usr/bin/env bash
# vllm branch: systalyze/dspark-gemma4-0.28.0
# Config 11 + EAGLE-3 speculation: Qwen3-32B on G7e (g7e.24xlarge, 4x RTX PRO 6000 96 GB) (NVFP4; DP8×TP1; FP8 KV cache; max-num-seqs 16; max-num-batched-tokens 16384)
# model /data/models/Qwen3-32B-NVFP4
# image vllm/vllm-openai:v0.28.0
# GPUs [0] (TP1: 1 replica(s))
# speculative decoding: eagle3, 3 draft tokens, draft /data/models/Qwen3-32B-speculator.eagle3
set -euo pipefail
BASE_PORT=${BASE_PORT:-8300}
# repository root, for the bind mounts (patched vLLM files from vllm/, the draft under models/)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

docker run -d --name aws-spec2-g7e-qwen-opt1-eagle3-k3-gpu0 --gpus '"device=0"' --ipc=host --shm-size=16g -p $((BASE_PORT+0)):$((BASE_PORT+0)) -v /data:/data -e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/data/fi-tactics vllm/vllm-openai:v0.28.0 --model /data/models/Qwen3-32B-NVFP4 --served-model-name spec2-g7e-qwen-opt1-eagle3-k3 --tensor-parallel-size 1 --port $((BASE_PORT+0)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --max-num-seqs 16 --max-num-batched-tokens 16384 --kv-cache-dtype fp8 --speculative-config '{"method": "eagle3", "model": "/data/models/Qwen3-32B-speculator.eagle3", "num_speculative_tokens": 3}'

# Stop ONLY these containers, by name:
#   docker stop aws-spec2-g7e-qwen-opt1-eagle3-k3-gpu0
