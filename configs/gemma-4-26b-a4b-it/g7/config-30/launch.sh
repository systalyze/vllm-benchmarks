#!/usr/bin/env bash
# vllm branch: systalyze/serving-0.27.1
# Config 30: Gemma-4-26B-A4B-IT on G7 (g7.48xlarge, 8x RTX PRO 4500 32 GB) (NVFP4; DP8×TP1; FP8 KV cache; max-num-seqs 16; max-num-batched-tokens 4096; FLASHINFER attention)
# model /data/models/gemma-4-26B-A4B-it-NVFP4
# image systalyze/vllm-openai:v0.27.1-tf5.14.1
# GPUs [0] (TP1: 1 replica(s))
# 1 patched vLLM file(s) from the vllm/ submodule (systalyze/serving-0.27.1) bind-mounted read-only over the stock image.
set -euo pipefail
BASE_PORT=${BASE_PORT:-8300}
# repository root, for the bind mounts (patched vLLM files from vllm/, the draft under models/)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

docker run -d --name aws-g7-gem-nvfp4-tp1-fp8kv-mns16-mnbt4096-flashinfer-gpu0 --gpus '"device=0"' --ipc=host --shm-size=16g -p $((BASE_PORT+0)):$((BASE_PORT+0)) -v /data:/data -e AUDIT_GEMMA4_NO_MM_PREFIX=1 -v $REPO/vllm/vllm/transformers_utils/model_arch_config_convertor.py:/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/model_arch_config_convertor.py:ro systalyze/vllm-openai:v0.27.1-tf5.14.1 --model /data/models/gemma-4-26B-A4B-it-NVFP4 --served-model-name g7-gem-nvfp4-tp1-fp8kv-mns16-mnbt4096-flashinfer --tensor-parallel-size 1 --port $((BASE_PORT+0)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --max-num-seqs 16 --max-num-batched-tokens 4096 --hf-overrides '{"architectures": ["Gemma4ForCausalLM"]}' --kv-cache-dtype fp8 --attention-backend FLASHINFER

# Stop ONLY these containers, by name:
#   docker stop aws-g7-gem-nvfp4-tp1-fp8kv-mns16-mnbt4096-flashinfer-gpu0
