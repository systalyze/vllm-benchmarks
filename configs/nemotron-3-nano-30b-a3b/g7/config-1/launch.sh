#!/usr/bin/env bash
# vllm branch: systalyze/serving-0.27.1
# Config 1: Nemotron-3-Nano-30B-A3B on G7 (g7.48xlarge, 8x RTX PRO 4500 32 GB) (NVFP4; DP8×TP1; BF16 KV cache; max-num-seqs 64; max-num-batched-tokens 16384)
# model /data/models/Nemotron-3-Nano-30B-A3B-NVFP4
# image vllm/vllm-openai:v0.27.1
# GPUs [0, 1, 2, 3, 4, 5, 6, 7] (TP1: 8 replica(s))
set -euo pipefail
BASE_PORT=${BASE_PORT:-8300}
# repository root, for the bind mounts (patched vLLM files from vllm/, the draft under models/)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

docker run -d --name aws-g7-nem-nvfp4-mns64-mnbt16384-gpu0 --gpus '"device=0"' --ipc=host --shm-size=16g -p $((BASE_PORT+0)):$((BASE_PORT+0)) -v /data:/data vllm/vllm-openai:v0.27.1 --model /data/models/Nemotron-3-Nano-30B-A3B-NVFP4 --served-model-name g7-nem-nvfp4-mns64-mnbt16384 --tensor-parallel-size 1 --port $((BASE_PORT+0)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --max-num-seqs 64 --max-num-batched-tokens 16384
docker run -d --name aws-g7-nem-nvfp4-mns64-mnbt16384-gpu1 --gpus '"device=1"' --ipc=host --shm-size=16g -p $((BASE_PORT+1)):$((BASE_PORT+1)) -v /data:/data vllm/vllm-openai:v0.27.1 --model /data/models/Nemotron-3-Nano-30B-A3B-NVFP4 --served-model-name g7-nem-nvfp4-mns64-mnbt16384 --tensor-parallel-size 1 --port $((BASE_PORT+1)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --max-num-seqs 64 --max-num-batched-tokens 16384
docker run -d --name aws-g7-nem-nvfp4-mns64-mnbt16384-gpu2 --gpus '"device=2"' --ipc=host --shm-size=16g -p $((BASE_PORT+2)):$((BASE_PORT+2)) -v /data:/data vllm/vllm-openai:v0.27.1 --model /data/models/Nemotron-3-Nano-30B-A3B-NVFP4 --served-model-name g7-nem-nvfp4-mns64-mnbt16384 --tensor-parallel-size 1 --port $((BASE_PORT+2)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --max-num-seqs 64 --max-num-batched-tokens 16384
docker run -d --name aws-g7-nem-nvfp4-mns64-mnbt16384-gpu3 --gpus '"device=3"' --ipc=host --shm-size=16g -p $((BASE_PORT+3)):$((BASE_PORT+3)) -v /data:/data vllm/vllm-openai:v0.27.1 --model /data/models/Nemotron-3-Nano-30B-A3B-NVFP4 --served-model-name g7-nem-nvfp4-mns64-mnbt16384 --tensor-parallel-size 1 --port $((BASE_PORT+3)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --max-num-seqs 64 --max-num-batched-tokens 16384
docker run -d --name aws-g7-nem-nvfp4-mns64-mnbt16384-gpu4 --gpus '"device=4"' --ipc=host --shm-size=16g -p $((BASE_PORT+4)):$((BASE_PORT+4)) -v /data:/data vllm/vllm-openai:v0.27.1 --model /data/models/Nemotron-3-Nano-30B-A3B-NVFP4 --served-model-name g7-nem-nvfp4-mns64-mnbt16384 --tensor-parallel-size 1 --port $((BASE_PORT+4)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --max-num-seqs 64 --max-num-batched-tokens 16384
docker run -d --name aws-g7-nem-nvfp4-mns64-mnbt16384-gpu5 --gpus '"device=5"' --ipc=host --shm-size=16g -p $((BASE_PORT+5)):$((BASE_PORT+5)) -v /data:/data vllm/vllm-openai:v0.27.1 --model /data/models/Nemotron-3-Nano-30B-A3B-NVFP4 --served-model-name g7-nem-nvfp4-mns64-mnbt16384 --tensor-parallel-size 1 --port $((BASE_PORT+5)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --max-num-seqs 64 --max-num-batched-tokens 16384
docker run -d --name aws-g7-nem-nvfp4-mns64-mnbt16384-gpu6 --gpus '"device=6"' --ipc=host --shm-size=16g -p $((BASE_PORT+6)):$((BASE_PORT+6)) -v /data:/data vllm/vllm-openai:v0.27.1 --model /data/models/Nemotron-3-Nano-30B-A3B-NVFP4 --served-model-name g7-nem-nvfp4-mns64-mnbt16384 --tensor-parallel-size 1 --port $((BASE_PORT+6)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --max-num-seqs 64 --max-num-batched-tokens 16384
docker run -d --name aws-g7-nem-nvfp4-mns64-mnbt16384-gpu7 --gpus '"device=7"' --ipc=host --shm-size=16g -p $((BASE_PORT+7)):$((BASE_PORT+7)) -v /data:/data vllm/vllm-openai:v0.27.1 --model /data/models/Nemotron-3-Nano-30B-A3B-NVFP4 --served-model-name g7-nem-nvfp4-mns64-mnbt16384 --tensor-parallel-size 1 --port $((BASE_PORT+7)) --host 0.0.0.0 --max-model-len 12288 --no-enable-prefix-caching --load-format auto --max-num-seqs 64 --max-num-batched-tokens 16384

# Stop ONLY these containers, by name:
#   docker stop aws-g7-nem-nvfp4-mns64-mnbt16384-gpu0
#   docker stop aws-g7-nem-nvfp4-mns64-mnbt16384-gpu1
#   docker stop aws-g7-nem-nvfp4-mns64-mnbt16384-gpu2
#   docker stop aws-g7-nem-nvfp4-mns64-mnbt16384-gpu3
#   docker stop aws-g7-nem-nvfp4-mns64-mnbt16384-gpu4
#   docker stop aws-g7-nem-nvfp4-mns64-mnbt16384-gpu5
#   docker stop aws-g7-nem-nvfp4-mns64-mnbt16384-gpu6
#   docker stop aws-g7-nem-nvfp4-mns64-mnbt16384-gpu7
