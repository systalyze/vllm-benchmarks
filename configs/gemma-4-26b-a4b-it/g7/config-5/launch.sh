#!/usr/bin/env bash
# vllm branch: systalyze/serving-0.27.1
# Config 5: Gemma-4-26B-A4B-IT on G7 (g7.48xlarge, 8x RTX PRO 4500 32 GB) (NVFP4; DP4×TP2 + EP; BF16 KV cache; max-num-seqs 64; max-num-batched-tokens 4096)
# model /data/models/gemma-4-26B-A4B-it-NVFP4
# image systalyze/vllm-openai:v0.27.1-tf5.14.1
# GPUs [0, 1, 2, 3, 4, 5, 6, 7] (TP2: 4 replica(s))
set -euo pipefail
BASE_PORT=${BASE_PORT:-8300}
# repository root, for the bind mounts (patched vLLM files from vllm/, the draft under models/)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

docker run -d --name aws-g7-gem-nvfp4-tp2-ep-gpu0 --gpus '"device=0,1"' --ipc=host --shm-size=16g -p $((BASE_PORT+0)):$((BASE_PORT+0)) -v /data:/data systalyze/vllm-openai:v0.27.1-tf5.14.1 --model /data/models/gemma-4-26B-A4B-it-NVFP4 --served-model-name g7-gem-nvfp4-tp2-ep --tensor-parallel-size 2 --port $((BASE_PORT+0)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --enable-expert-parallel --max-num-seqs 64 --max-num-batched-tokens 4096 --hf-overrides '{"architectures": ["Gemma4ForCausalLM"]}'
docker run -d --name aws-g7-gem-nvfp4-tp2-ep-gpu2 --gpus '"device=2,3"' --ipc=host --shm-size=16g -p $((BASE_PORT+1)):$((BASE_PORT+1)) -v /data:/data systalyze/vllm-openai:v0.27.1-tf5.14.1 --model /data/models/gemma-4-26B-A4B-it-NVFP4 --served-model-name g7-gem-nvfp4-tp2-ep --tensor-parallel-size 2 --port $((BASE_PORT+1)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --enable-expert-parallel --max-num-seqs 64 --max-num-batched-tokens 4096 --hf-overrides '{"architectures": ["Gemma4ForCausalLM"]}'
docker run -d --name aws-g7-gem-nvfp4-tp2-ep-gpu4 --gpus '"device=4,5"' --ipc=host --shm-size=16g -p $((BASE_PORT+2)):$((BASE_PORT+2)) -v /data:/data systalyze/vllm-openai:v0.27.1-tf5.14.1 --model /data/models/gemma-4-26B-A4B-it-NVFP4 --served-model-name g7-gem-nvfp4-tp2-ep --tensor-parallel-size 2 --port $((BASE_PORT+2)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --enable-expert-parallel --max-num-seqs 64 --max-num-batched-tokens 4096 --hf-overrides '{"architectures": ["Gemma4ForCausalLM"]}'
docker run -d --name aws-g7-gem-nvfp4-tp2-ep-gpu6 --gpus '"device=6,7"' --ipc=host --shm-size=16g -p $((BASE_PORT+3)):$((BASE_PORT+3)) -v /data:/data systalyze/vllm-openai:v0.27.1-tf5.14.1 --model /data/models/gemma-4-26B-A4B-it-NVFP4 --served-model-name g7-gem-nvfp4-tp2-ep --tensor-parallel-size 2 --port $((BASE_PORT+3)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --enable-expert-parallel --max-num-seqs 64 --max-num-batched-tokens 4096 --hf-overrides '{"architectures": ["Gemma4ForCausalLM"]}'

# Stop ONLY these containers, by name:
#   docker stop aws-g7-gem-nvfp4-tp2-ep-gpu0
#   docker stop aws-g7-gem-nvfp4-tp2-ep-gpu2
#   docker stop aws-g7-gem-nvfp4-tp2-ep-gpu4
#   docker stop aws-g7-gem-nvfp4-tp2-ep-gpu6
