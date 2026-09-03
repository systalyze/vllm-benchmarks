#!/usr/bin/env bash
# vllm branch: systalyze/serving-0.27.1
# Config 1: Qwen3-32B on G7 (g7.48xlarge, 8x RTX PRO 4500 32 GB) (NVFP4; DP4×TP2; FP8 KV cache; max-num-seqs 64; max-num-batched-tokens 8192)
# model /data/models/Qwen3-32B-NVFP4
# image vllm/vllm-openai:v0.27.1
# GPUs [4, 5] (TP2: 1 replica(s))
set -euo pipefail
BASE_PORT=${BASE_PORT:-8300}
# repository root, for the bind mounts (patched vLLM files from vllm/, the draft under models/)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

docker run -d --name aws-g7-qwen-nvfp4-tp2-fp8kv-tunedcache-gpu4 --gpus '"device=4,5"' --ipc=host --shm-size=16g -p $((BASE_PORT+0)):$((BASE_PORT+0)) -v /data:/data -e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/data/fi-tactics-lowmem vllm/vllm-openai:v0.27.1 --model /data/models/Qwen3-32B-NVFP4 --served-model-name g7-qwen-nvfp4-tp2-fp8kv-tunedcache --tensor-parallel-size 2 --port $((BASE_PORT+0)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --max-num-seqs 64 --max-num-batched-tokens 8192 --kv-cache-dtype fp8

# Stop ONLY these containers, by name:
#   docker stop aws-g7-qwen-nvfp4-tp2-fp8kv-tunedcache-gpu4
