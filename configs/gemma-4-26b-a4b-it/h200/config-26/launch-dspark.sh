#!/usr/bin/env bash
# vllm branch: systalyze/dspark-gemma4-0.28.0
# Config 26 + DSpark speculation: Gemma-4-26B-A4B-IT on H200 (p5en.48xlarge, 8x H200 SXM 141 GB) (FP8-dynamic; DP8×TP1; BF16 KV cache; max-num-seqs 8; max-num-batched-tokens 16384)
# model /data/models/gemma-4-26B-A4B-it-FP8-dynamic
# image vllm/vllm-openai:v0.28.0
# GPUs [0] (TP1: 1 replica(s))
# 1 patched vLLM file(s) from the vllm/ submodule (systalyze/dspark-gemma4-0.28.0) bind-mounted read-only over the stock image.
# speculative decoding: dspark, 5 draft tokens, draft /data/models/dspark-gemma4-26b-a4b
set -euo pipefail
BASE_PORT=${BASE_PORT:-8300}
# repository root, for the bind mounts (patched vLLM files from vllm/)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

docker run -d --name aws-spec-h200g-gem-dspark-ours-k5-fp8-mns8-mnbt16384-gpu0 --gpus '"device=0"' --ipc=host --shm-size=16g -p $((BASE_PORT+0)):$((BASE_PORT+0)) -v /data:/data -v $REPO/vllm/vllm/model_executor/models/gemma4_dspark.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4_dspark.py:ro vllm/vllm-openai:v0.28.0 --model /data/models/gemma-4-26B-A4B-it-FP8-dynamic --served-model-name spec-h200g-gem-dspark-ours-k5-fp8-mns8-mnbt16384 --tensor-parallel-size 1 --port $((BASE_PORT+0)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --hf-overrides '{"architectures": ["Gemma4ForCausalLM"]}' --max-num-batched-tokens 16384 --max-num-seqs 8 --kv-cache-dtype auto --trust-remote-code --speculative-config "{\"method\": \"dspark\", \"model\": \"/data/models/dspark-gemma4-26b-a4b\", \"num_speculative_tokens\": 5, \"draft_sample_method\": \"greedy\"}"

# Stop ONLY these containers, by name:
#   docker stop aws-spec-h200g-gem-dspark-ours-k5-fp8-mns8-mnbt16384-gpu0
