#!/usr/bin/env bash
# vllm branch: systalyze/serving-0.27.1
# Config 22: Qwen3-32B on H200 (p5en.48xlarge, 8x H200 SXM 141 GB) (FP8; DP8×TP1; FP8 KV cache; max-num-seqs 8; max-num-batched-tokens 16384)
# model /data/models/Qwen3-32B-FP8
# image vllm/vllm-openai:v0.27.1
# GPUs [0] (TP1: 1 replica(s))
set -euo pipefail
BASE_PORT=${BASE_PORT:-8300}
# repository root, for the bind mounts (patched vLLM files from vllm/, the draft under models/)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

docker run -d --name aws-h200g-qwen-fp8-fp8kv-mns8-mnbt16384-gpu0 --gpus '"device=0"' --ipc=host --shm-size=16g -p $((BASE_PORT+0)):$((BASE_PORT+0)) -v /data:/data vllm/vllm-openai:v0.27.1 --model /data/models/Qwen3-32B-FP8 --served-model-name h200g-qwen-fp8-fp8kv-mns8-mnbt16384 --tensor-parallel-size 1 --port $((BASE_PORT+0)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --max-num-batched-tokens 16384 --max-num-seqs 8

# Stop ONLY these containers, by name:
#   docker stop aws-h200g-qwen-fp8-fp8kv-mns8-mnbt16384-gpu0
