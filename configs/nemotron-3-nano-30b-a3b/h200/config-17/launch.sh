#!/usr/bin/env bash
# vllm branch: systalyze/dspark-gemma4-0.28.0
# Config 17: Nemotron-3-Nano-30B-A3B on H200 (p5en.48xlarge, 8x H200 SXM 141 GB) (FP8; DP8×TP1; FP8 KV cache; max-num-seqs 256; max-num-batched-tokens 16384)
# model /data/models/Nemotron-3-Nano-30B-A3B-FP8
# image vllm/vllm-openai:v0.28.0
# GPUs [0] (TP1: 1 replica(s))
set -euo pipefail
BASE_PORT=${BASE_PORT:-8300}
# repository root, for the bind mounts (patched vLLM files from vllm/, the draft under models/)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

docker run -d --name aws-h200g-nem-fp8-v0280-r512-gpu0 --gpus '"device=0"' --ipc=host --shm-size=16g -p $((BASE_PORT+0)):$((BASE_PORT+0)) -v /data:/data vllm/vllm-openai:v0.28.0 --model /data/models/Nemotron-3-Nano-30B-A3B-FP8 --served-model-name h200g-nem-fp8-v0280-r512 --tensor-parallel-size 1 --port $((BASE_PORT+0)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --max-num-batched-tokens 16384 --max-num-seqs 256 --hf-overrides '{"chunk_size": 256}' --compilation-config '{"custom_ops": ["+mixer2_gated_rms_norm"]}'

# Stop ONLY these containers, by name:
#   docker stop aws-h200g-nem-fp8-v0280-r512-gpu0
