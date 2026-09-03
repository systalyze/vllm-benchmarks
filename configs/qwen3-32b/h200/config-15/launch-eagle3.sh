#!/usr/bin/env bash
# vllm branch: systalyze/dspark-gemma4-0.28.0
# Config 15 + EAGLE-3 speculation: Qwen3-32B on H200 (p5en.48xlarge, 8x H200 SXM 141 GB) (FP8; DP8×TP1; FP8 KV cache; max-num-seqs 16; max-num-batched-tokens 8192)
# model /data/models/Qwen3-32B-FP8
# image vllm/vllm-openai:v0.28.0
# GPUs [0] (TP1: 1 replica(s))
# speculative decoding: eagle3, 3 draft tokens, draft /data/models/Qwen3-32B-speculator.eagle3
set -euo pipefail
BASE_PORT=${BASE_PORT:-8300}
# repository root, for the bind mounts (patched vLLM files from vllm/, the draft under models/)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

docker run -d --name aws-spec-h200g-qwen-eagle3-k3-fp8kv-mns16-mnbt8192-gpu0 --gpus '"device=0"' --ipc=host --shm-size=16g -p $((BASE_PORT+0)):$((BASE_PORT+0)) -v /data:/data vllm/vllm-openai:v0.28.0 --model /data/models/Qwen3-32B-FP8 --served-model-name spec-h200g-qwen-eagle3-k3-fp8kv-mns16-mnbt8192 --tensor-parallel-size 1 --port $((BASE_PORT+0)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --max-num-batched-tokens 8192 --max-num-seqs 16 --speculative-config '{"method": "eagle3", "model": "/data/models/Qwen3-32B-speculator.eagle3", "num_speculative_tokens": 3}'

# Stop ONLY these containers, by name:
#   docker stop aws-spec-h200g-qwen-eagle3-k3-fp8kv-mns16-mnbt8192-gpu0
