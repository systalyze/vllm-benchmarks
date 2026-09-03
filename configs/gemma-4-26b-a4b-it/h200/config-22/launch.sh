#!/usr/bin/env bash
# vllm branch: systalyze/serving-0.27.1
# Config 22: Gemma-4-26B-A4B-IT on H200 (p5en.48xlarge, 8x H200 SXM 141 GB) (FP8-dynamic; DP4×TP2; BF16 KV cache; max-num-seqs 256; max-num-batched-tokens 16384)
# model /data/models/gemma-4-26B-A4B-it-FP8-dynamic
# image systalyze/vllm-openai:v0.27.1-tf5.14.1
# GPUs [0, 1, 2, 3, 4, 5, 6, 7] (TP2: 4 replica(s))
set -euo pipefail
BASE_PORT=${BASE_PORT:-8300}
# repository root, for the bind mounts (patched vLLM files from vllm/, the draft under models/)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

docker run -d --name aws-h200-gem-fp8-tp2-gpu0 --gpus '"device=0,1"' --ipc=host --shm-size=16g -p $((BASE_PORT+0)):$((BASE_PORT+0)) -v /data:/data systalyze/vllm-openai:v0.27.1-tf5.14.1 --model /data/models/gemma-4-26B-A4B-it-FP8-dynamic --served-model-name h200-gem-fp8-tp2 --tensor-parallel-size 2 --port $((BASE_PORT+0)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --hf-overrides '{"architectures": ["Gemma4ForCausalLM"]}' --max-num-batched-tokens 16384 --max-num-seqs 256 --gpu-memory-utilization 0.90
docker run -d --name aws-h200-gem-fp8-tp2-gpu2 --gpus '"device=2,3"' --ipc=host --shm-size=16g -p $((BASE_PORT+1)):$((BASE_PORT+1)) -v /data:/data systalyze/vllm-openai:v0.27.1-tf5.14.1 --model /data/models/gemma-4-26B-A4B-it-FP8-dynamic --served-model-name h200-gem-fp8-tp2 --tensor-parallel-size 2 --port $((BASE_PORT+1)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --hf-overrides '{"architectures": ["Gemma4ForCausalLM"]}' --max-num-batched-tokens 16384 --max-num-seqs 256 --gpu-memory-utilization 0.90
docker run -d --name aws-h200-gem-fp8-tp2-gpu4 --gpus '"device=4,5"' --ipc=host --shm-size=16g -p $((BASE_PORT+2)):$((BASE_PORT+2)) -v /data:/data systalyze/vllm-openai:v0.27.1-tf5.14.1 --model /data/models/gemma-4-26B-A4B-it-FP8-dynamic --served-model-name h200-gem-fp8-tp2 --tensor-parallel-size 2 --port $((BASE_PORT+2)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --hf-overrides '{"architectures": ["Gemma4ForCausalLM"]}' --max-num-batched-tokens 16384 --max-num-seqs 256 --gpu-memory-utilization 0.90
docker run -d --name aws-h200-gem-fp8-tp2-gpu6 --gpus '"device=6,7"' --ipc=host --shm-size=16g -p $((BASE_PORT+3)):$((BASE_PORT+3)) -v /data:/data systalyze/vllm-openai:v0.27.1-tf5.14.1 --model /data/models/gemma-4-26B-A4B-it-FP8-dynamic --served-model-name h200-gem-fp8-tp2 --tensor-parallel-size 2 --port $((BASE_PORT+3)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --hf-overrides '{"architectures": ["Gemma4ForCausalLM"]}' --max-num-batched-tokens 16384 --max-num-seqs 256 --gpu-memory-utilization 0.90

# Stop ONLY these containers, by name:
#   docker stop aws-h200-gem-fp8-tp2-gpu0
#   docker stop aws-h200-gem-fp8-tp2-gpu2
#   docker stop aws-h200-gem-fp8-tp2-gpu4
#   docker stop aws-h200-gem-fp8-tp2-gpu6
