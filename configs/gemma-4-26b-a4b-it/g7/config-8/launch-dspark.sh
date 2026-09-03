#!/usr/bin/env bash
# vllm branch: systalyze/dspark-gemma4-0.28.0
# Config 8 + DSpark speculation: Gemma-4-26B-A4B-IT on G7 (g7.48xlarge, 8x RTX PRO 4500 32 GB) (NVFP4; DP4×TP2 + EP; BF16 KV cache; max-num-seqs 16; max-num-batched-tokens 4096)
# model /data/models/gemma-4-26B-A4B-it-NVFP4
# image vllm/vllm-openai:v0.28.0
# GPUs [0, 1] (TP2: 1 replica(s))
# 1 patched vLLM file(s) from the vllm/ submodule (systalyze/dspark-gemma4-0.28.0) bind-mounted read-only over the stock image.
# speculative decoding: dspark, 5 draft tokens, draft /data/models/dspark-gemma4-26b-a4b
set -euo pipefail
BASE_PORT=${BASE_PORT:-8300}
# repository root, for the bind mounts (patched vLLM files from vllm/)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

docker run -d --name aws-spec2-g7-gem-opt8-dspark-k5-gpu0 --gpus '"device=0,1"' --ipc=host --shm-size=16g -p $((BASE_PORT+0)):$((BASE_PORT+0)) -v /data:/data -v $REPO/vllm/vllm/model_executor/models/gemma4_dspark.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4_dspark.py:ro vllm/vllm-openai:v0.28.0 --model /data/models/gemma-4-26B-A4B-it-NVFP4 --served-model-name spec2-g7-gem-opt8-dspark-k5 --tensor-parallel-size 2 --port $((BASE_PORT+0)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --enable-expert-parallel --max-num-seqs 16 --max-num-batched-tokens 4096 --hf-overrides '{"architectures": ["Gemma4ForCausalLM"]}' --speculative-config "{\"method\": \"dspark\", \"model\": \"/data/models/dspark-gemma4-26b-a4b\", \"num_speculative_tokens\": 5, \"draft_sample_method\": \"greedy\"}" --trust-remote-code

# Stop ONLY these containers, by name:
#   docker stop aws-spec2-g7-gem-opt8-dspark-k5-gpu0
