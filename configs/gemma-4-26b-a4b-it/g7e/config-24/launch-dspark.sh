#!/usr/bin/env bash
# vllm branch: systalyze/dspark-gemma4-0.28.0
# Config 24 + DSpark speculation: Gemma-4-26B-A4B-IT on G7e (g7e.24xlarge, 4x RTX PRO 6000 96 GB) (NVFP4; DP8×TP1; FP8 KV cache; max-num-seqs 256; max-num-batched-tokens 16384)
# model /data/models/gemma-4-26B-A4B-it-NVFP4
# image vllm/vllm-openai:v0.28.0
# GPUs [0] (TP1: 1 replica(s))
# 1 patched vLLM file(s) from the vllm/ submodule (systalyze/dspark-gemma4-0.28.0) bind-mounted read-only over the stock image.
# speculative decoding: dspark, 5 draft tokens, draft /data/models/dspark-gemma4-26b-a4b
set -euo pipefail
BASE_PORT=${BASE_PORT:-8300}
# repository root, for the bind mounts (patched vLLM files from vllm/)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

docker run -d --name aws-spec2-g7e-gem-mns256-dspark-k5-gpu0 --gpus '"device=0"' --ipc=host --shm-size=16g -p $((BASE_PORT+0)):$((BASE_PORT+0)) -v /data:/data -e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/data/fi-tactics -v $REPO/vllm/vllm/model_executor/models/gemma4_dspark.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4_dspark.py:ro vllm/vllm-openai:v0.28.0 --model /data/models/gemma-4-26B-A4B-it-NVFP4 --served-model-name spec2-g7e-gem-mns256-dspark-k5 --tensor-parallel-size 1 --port $((BASE_PORT+0)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --max-num-seqs 256 --max-num-batched-tokens 16384 --hf-overrides '{"architectures": ["Gemma4ForCausalLM"]}' --kv-cache-dtype fp8 --speculative-config "{\"method\": \"dspark\", \"model\": \"/data/models/dspark-gemma4-26b-a4b\", \"num_speculative_tokens\": 5, \"draft_sample_method\": \"greedy\"}" --trust-remote-code

# Stop ONLY these containers, by name:
#   docker stop aws-spec2-g7e-gem-mns256-dspark-k5-gpu0
