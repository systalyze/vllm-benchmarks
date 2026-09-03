#!/usr/bin/env bash
# vllm branch: systalyze/serving-0.27.1
# Config 23: Gemma-4-26B-A4B-IT on H200 (p5en.48xlarge, 8x H200 SXM 141 GB) (FP8-dynamic; DP2×TP4; BF16 KV cache; max-num-seqs 256; max-num-batched-tokens 16384)
# model /data/models/gemma-4-26B-A4B-it-FP8-dynamic
# image systalyze/vllm-openai:v0.27.1-tf5.14.1
# GPUs [1, 2, 3, 7] (TP4: 1 replica(s))
set -euo pipefail
BASE_PORT=${BASE_PORT:-8300}
# repository root, for the bind mounts (patched vLLM files from vllm/, the draft under models/)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

docker run -d --name aws-h200-gem-fp8-tp4-gpu1 --gpus '"device=1,2,3,7"' --ipc=host --shm-size=16g -p $((BASE_PORT+0)):$((BASE_PORT+0)) -v /data:/data systalyze/vllm-openai:v0.27.1-tf5.14.1 --model /data/models/gemma-4-26B-A4B-it-FP8-dynamic --served-model-name h200-gem-fp8-tp4 --tensor-parallel-size 4 --port $((BASE_PORT+0)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --hf-overrides '{"architectures": ["Gemma4ForCausalLM"]}' --max-num-batched-tokens 16384 --max-num-seqs 256 --kv-cache-dtype auto

# Stop ONLY these containers, by name:
#   docker stop aws-h200-gem-fp8-tp4-gpu1
