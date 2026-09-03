# Configurations

One folder per configuration on the frontier, plus the stock BF16 baseline of each
(model, instance) pair. Each folder holds `launch.sh` (the exact `docker run` line) and `row.json`
(the same deployment as data, which `bench/run.sh` drives). Where the speculative-decoding arm of a
configuration was measured, the folder also holds `launch-dspark.sh` / `row-dspark.json` (Gemma, our
DSpark draft) or `launch-eagle3.sh` / `row-eagle3.json` (Qwen, EAGLE-3); run it with
`bench/run.sh <folder> --arm dspark` or `--arm eagle3`. Config numbers are shared across a model's three instances
(numbered per model, so the same number on two instances is two different configurations).

Arrangement: DPn×TPm = n replicas of m GPUs each on the full node; EP = expert parallelism.

## Gemma-4-26B-A4B-IT

| Config | Instance | Configuration | Speculation | Folder |
|---|---|---|---|---|
| baseline | G7 | BF16; DP2×TP4; stock flags (only --max-model-len 12288 and no prefix caching); vLLM 0.27.1 + transformers 5.14.1 | none | `configs/gemma-4-26b-a4b-it/g7/baseline/` |
| Config 1 | G7 | NVFP4; DP8×TP1; FP8 KV cache; max-num-seqs 64; max-num-batched-tokens 4096; vLLM 0.27.1 + transformers 5.14.1 | DSpark k=5, measured on this config (launch-dspark.sh) | `configs/gemma-4-26b-a4b-it/g7/config-1/` |
| Config 5 | G7 | NVFP4; DP4×TP2 + EP; BF16 KV cache; max-num-seqs 64; max-num-batched-tokens 4096; vLLM 0.27.1 + transformers 5.14.1 | DSpark k=5, measured on this config (launch-dspark.sh) | `configs/gemma-4-26b-a4b-it/g7/config-5/` |
| Config 6 | G7 | NVFP4; DP8×TP1; BF16 KV cache; max-num-seqs 8; max-num-batched-tokens 4096; FLASHINFER attention; vLLM 0.27.1 + transformers 5.14.1 | none (DSpark arm refused: the draft head_dim 512 is unsupported by FlashInfer on vLLM 0.28.0) | `configs/gemma-4-26b-a4b-it/g7/config-6/` |
| Config 7 | G7 | NVFP4; DP4×TP2 + EP; BF16 KV cache; max-num-seqs 32; max-num-batched-tokens 4096; vLLM 0.27.1 + transformers 5.14.1 | DSpark k=5, measured on this config (launch-dspark.sh) | `configs/gemma-4-26b-a4b-it/g7/config-7/` |
| Config 8 | G7 | NVFP4; DP4×TP2 + EP; BF16 KV cache; max-num-seqs 16; max-num-batched-tokens 4096; vLLM 0.27.1 + transformers 5.14.1 | DSpark k=5, measured on this config (launch-dspark.sh) | `configs/gemma-4-26b-a4b-it/g7/config-8/` |
| Config 9 | G7 | NVFP4; DP8×TP1; BF16 KV cache; max-num-seqs 4; max-num-batched-tokens 4096; vLLM 0.27.1 + transformers 5.14.1 | DSpark k=5, measured on this config (launch-dspark.sh) | `configs/gemma-4-26b-a4b-it/g7/config-9/` |
| Config 10 | G7 | NVFP4; DP4×TP2 + EP; BF16 KV cache; max-num-seqs 8; max-num-batched-tokens 4096; vLLM 0.27.1 + transformers 5.14.1 | DSpark k=5, measured on this config (launch-dspark.sh) | `configs/gemma-4-26b-a4b-it/g7/config-10/` |
| Config 29 | G7 | NVFP4; DP8×TP1; FP8 KV cache; max-num-seqs 32; max-num-batched-tokens 4096; FLASHINFER attention; vLLM 0.27.1 + transformers 5.14.1 | none (no arm: attention backend differs from the measured pair) | `configs/gemma-4-26b-a4b-it/g7/config-29/` |
| Config 30 | G7 | NVFP4; DP8×TP1; FP8 KV cache; max-num-seqs 16; max-num-batched-tokens 4096; FLASHINFER attention; vLLM 0.27.1 + transformers 5.14.1 | none (no arm: attention backend differs from the measured pair) | `configs/gemma-4-26b-a4b-it/g7/config-30/` |
| Config 31 | G7 | NVFP4; DP8×TP1; FP8 KV cache; max-num-seqs 64; max-num-batched-tokens 4096; FLASHINFER attention; vLLM 0.27.1 + transformers 5.14.1 | none (DSpark arm refused: the draft head_dim 512 is unsupported by FlashInfer on vLLM 0.28.0) | `configs/gemma-4-26b-a4b-it/g7/config-31/` |
| baseline | G7e | BF16; one TP1 replica per GPU; stock flags (only --max-model-len 12288 and no prefix caching); vLLM 0.27.1 + transformers 5.14.1 | none | `configs/gemma-4-26b-a4b-it/g7e/baseline/` |
| Config 12 | G7e | NVFP4; DP8×TP1; FP8 KV cache; max-num-seqs 64; max-num-batched-tokens 8192; vLLM 0.27.1 + transformers 5.14.1 | DSpark k=5, measured on this config (launch-dspark.sh) | `configs/gemma-4-26b-a4b-it/g7e/config-12/` |
| Config 14 | G7e | NVFP4; DP8×TP1; FP8 KV cache; max-num-seqs 32; max-num-batched-tokens 16384; vLLM 0.27.1 + transformers 5.14.1 | DSpark k=5, measured on this config (launch-dspark.sh) | `configs/gemma-4-26b-a4b-it/g7e/config-14/` |
| Config 15 | G7e | NVFP4; DP8×TP1; FP8 KV cache; max-num-seqs 16; max-num-batched-tokens 16384; vLLM 0.27.1 + transformers 5.14.1 | DSpark k=5, measured on this config (launch-dspark.sh) | `configs/gemma-4-26b-a4b-it/g7e/config-15/` |
| Config 16 | G7e | NVFP4; DP8×TP1; FP8 KV cache; max-num-seqs 8; max-num-batched-tokens 16384; FLASHINFER attention; vLLM 0.27.1 + transformers 5.14.1 | none (DSpark arm refused: the draft head_dim 512 is unsupported by FlashInfer on vLLM 0.28.0) | `configs/gemma-4-26b-a4b-it/g7e/config-16/` |
| Config 17 | G7e | NVFP4; DP8×TP1; FP8 KV cache; max-num-seqs 4; max-num-batched-tokens 16384; vLLM 0.27.1 + transformers 5.14.1 | DSpark k=5, measured on this config (launch-dspark.sh) | `configs/gemma-4-26b-a4b-it/g7e/config-17/` |
| Config 24 | G7e | NVFP4; DP8×TP1; FP8 KV cache; max-num-seqs 256; max-num-batched-tokens 16384; vLLM 0.27.1 + transformers 5.14.1 | DSpark k=5, measured on this config (launch-dspark.sh) | `configs/gemma-4-26b-a4b-it/g7e/config-24/` |
| Config 25 | G7e | NVFP4; DP8×TP1; FP8 KV cache; max-num-seqs 128; max-num-batched-tokens 16384; FLASHINFER attention; vLLM 0.27.1 + transformers 5.14.1 | none (DSpark arm refused: the draft head_dim 512 is unsupported by FlashInfer on vLLM 0.28.0) | `configs/gemma-4-26b-a4b-it/g7e/config-25/` |
| baseline | H200 | BF16; one TP1 replica per GPU; stock flags (only --max-model-len 12288 and no prefix caching); vLLM 0.27.1 + transformers 5.14.1 | none | `configs/gemma-4-26b-a4b-it/h200/baseline/` |
| Config 19 | H200 | FP8-dynamic; DP8×TP1; BF16 KV cache; max-num-seqs 256; max-num-batched-tokens 8192; vLLM 0.27.1 + transformers 5.14.1 | DSpark k=5, measured on this config (launch-dspark.sh) | `configs/gemma-4-26b-a4b-it/h200/config-19/` |
| Config 20 | H200 | mixed-w4a16-fp8; DP8×TP1; BF16 KV cache; max-num-seqs 256; max-num-batched-tokens 16384; vLLM 0.27.1 + transformers 5.14.1 | DSpark k=5, measured on this config (launch-dspark.sh) | `configs/gemma-4-26b-a4b-it/h200/config-20/` |
| Config 21 | H200 | FP8-dynamic; DP8×TP1; BF16 KV cache; max-num-seqs 16; max-num-batched-tokens 8192; vLLM 0.27.1 + transformers 5.14.1 | DSpark k=5, measured on this config (launch-dspark.sh) | `configs/gemma-4-26b-a4b-it/h200/config-21/` |
| Config 22 | H200 | FP8-dynamic; DP4×TP2; BF16 KV cache; max-num-seqs 256; max-num-batched-tokens 16384; vLLM 0.27.1 + transformers 5.14.1 | DSpark k=5, measured on this config (launch-dspark.sh) | `configs/gemma-4-26b-a4b-it/h200/config-22/` |
| Config 23 | H200 | FP8-dynamic; DP2×TP4; BF16 KV cache; max-num-seqs 256; max-num-batched-tokens 16384; vLLM 0.27.1 + transformers 5.14.1 | none (DSpark arm refused: the draft does not construct at TP4 on vLLM 0.28.0) | `configs/gemma-4-26b-a4b-it/h200/config-23/` |
| Config 26 | H200 | FP8-dynamic; DP8×TP1; BF16 KV cache; max-num-seqs 8; max-num-batched-tokens 16384; vLLM 0.27.1 + transformers 5.14.1 | DSpark k=5, measured on this config (launch-dspark.sh) | `configs/gemma-4-26b-a4b-it/h200/config-26/` |
| Config 27 | H200 | FP8-dynamic; DP8×TP1; BF16 KV cache; max-num-seqs 4; max-num-batched-tokens 16384; vLLM 0.27.1 + transformers 5.14.1 | DSpark k=5, measured on this config (launch-dspark.sh) | `configs/gemma-4-26b-a4b-it/h200/config-27/` |
| Config 32 | H200 | FP8-dynamic; DP8×TP1; BF16 KV cache; max-num-seqs 256; max-num-batched-tokens 16384; vLLM 0.27.1 + transformers 5.14.1 | DSpark k=5, borrowed from Config 19's arm (not re-measured here) | `configs/gemma-4-26b-a4b-it/h200/config-32/` |

## Nemotron-3-Nano-30B-A3B

| Config | Instance | Configuration | Speculation | Folder |
|---|---|---|---|---|
| baseline | G7 | BF16; DP2×TP4; stock flags (only --max-model-len 12288 and no prefix caching); vLLM 0.27.1 | none | `configs/nemotron-3-nano-30b-a3b/g7/baseline/` |
| Config 1 | G7 | NVFP4; DP8×TP1; BF16 KV cache; max-num-seqs 64; max-num-batched-tokens 16384; vLLM 0.27.1 | none | `configs/nemotron-3-nano-30b-a3b/g7/config-1/` |
| Config 2 | G7 | NVFP4; DP8×TP1; BF16 KV cache; max-num-seqs 32; max-num-batched-tokens 8192; vLLM 0.27.1 | none | `configs/nemotron-3-nano-30b-a3b/g7/config-2/` |
| Config 3 | G7 | NVFP4; DP8×TP1; BF16 KV cache; max-num-seqs 16; max-num-batched-tokens 8192; vLLM 0.28.0 | none | `configs/nemotron-3-nano-30b-a3b/g7/config-3/` |
| Config 4 | G7 | NVFP4; DP4×TP2; BF16 KV cache; max-num-seqs 64; max-num-batched-tokens 8192; vLLM 0.27.1 | none | `configs/nemotron-3-nano-30b-a3b/g7/config-4/` |
| Config 5 | G7 | NVFP4; DP4×TP2; BF16 KV cache; max-num-seqs 32; max-num-batched-tokens 8192; vLLM 0.27.1 | none | `configs/nemotron-3-nano-30b-a3b/g7/config-5/` |
| Config 6 | G7 | NVFP4; DP8×TP1; BF16 KV cache; max-num-seqs 8; max-num-batched-tokens 8192; vLLM 0.27.1 | none | `configs/nemotron-3-nano-30b-a3b/g7/config-6/` |
| Config 7 | G7 | NVFP4; DP4×TP2; BF16 KV cache; max-num-seqs 16; max-num-batched-tokens 8192; vLLM 0.27.1 | none | `configs/nemotron-3-nano-30b-a3b/g7/config-7/` |
| Config 8 | G7 | NVFP4; DP8×TP1; BF16 KV cache; max-num-seqs 4; max-num-batched-tokens 8192; vLLM 0.27.1 | none | `configs/nemotron-3-nano-30b-a3b/g7/config-8/` |
| Config 9 | G7 | NVFP4; DP4×TP2; BF16 KV cache; max-num-seqs 8; max-num-batched-tokens 8192; vLLM 0.27.1 | none | `configs/nemotron-3-nano-30b-a3b/g7/config-9/` |
| baseline | G7e | BF16; one TP1 replica per GPU; stock flags (only --max-model-len 12288 and no prefix caching); vLLM 0.27.1 | none | `configs/nemotron-3-nano-30b-a3b/g7e/baseline/` |
| Config 10 | G7e | NVFP4; DP8×TP1; BF16 KV cache; max-num-seqs 128; max-num-batched-tokens 8192; vLLM 0.27.1 | none | `configs/nemotron-3-nano-30b-a3b/g7e/config-10/` |
| Config 11 | G7e | NVFP4; DP8×TP1; BF16 KV cache; max-num-seqs 128; max-num-batched-tokens 16384; vLLM 0.28.0 | none | `configs/nemotron-3-nano-30b-a3b/g7e/config-11/` |
| Config 12 | G7e | NVFP4; DP8×TP1; BF16 KV cache; max-num-seqs 32; max-num-batched-tokens 16384; vLLM 0.28.0 | none | `configs/nemotron-3-nano-30b-a3b/g7e/config-12/` |
| Config 13 | G7e | NVFP4; DP8×TP1; BF16 KV cache; max-num-seqs 16; max-num-batched-tokens 16384; vLLM 0.27.1 | none | `configs/nemotron-3-nano-30b-a3b/g7e/config-13/` |
| Config 14 | G7e | NVFP4; DP8×TP1; BF16 KV cache; max-num-seqs 8; max-num-batched-tokens 16384; vLLM 0.27.1 | none | `configs/nemotron-3-nano-30b-a3b/g7e/config-14/` |
| Config 15 | G7e | NVFP4; DP4×TP2; BF16 KV cache; max-num-seqs 32; max-num-batched-tokens 16384; vLLM 0.27.1 | none | `configs/nemotron-3-nano-30b-a3b/g7e/config-15/` |
| Config 16 | G7e | NVFP4; DP8×TP1; BF16 KV cache; max-num-seqs 4; max-num-batched-tokens 16384; vLLM 0.27.1 | none | `configs/nemotron-3-nano-30b-a3b/g7e/config-16/` |
| baseline | H200 | BF16; one TP1 replica per GPU; stock flags (only --max-model-len 12288 and no prefix caching); vLLM 0.27.1 | none | `configs/nemotron-3-nano-30b-a3b/h200/baseline/` |
| Config 17 | H200 | FP8; DP8×TP1; FP8 KV cache; max-num-seqs 256; max-num-batched-tokens 16384; vLLM 0.28.0 | none | `configs/nemotron-3-nano-30b-a3b/h200/config-17/` |
| Config 18 | H200 | FP8; DP4×TP2; FP8 KV cache; max-num-seqs 256; max-num-batched-tokens 16384; vLLM 0.27.1 | none | `configs/nemotron-3-nano-30b-a3b/h200/config-18/` |
| Config 19 | H200 | FP8; DP2×TP4; FP8 KV cache; max-num-seqs 256; max-num-batched-tokens 16384; vLLM 0.27.1 | none | `configs/nemotron-3-nano-30b-a3b/h200/config-19/` |
| Config 20 | H200 | FP8; DP8×TP1; FP8 KV cache; max-num-seqs 128; max-num-batched-tokens 16384; vLLM 0.27.1 | none | `configs/nemotron-3-nano-30b-a3b/h200/config-20/` |
| Config 21 | H200 | FP8; DP8×TP1; FP8 KV cache; max-num-seqs 8; max-num-batched-tokens 16384; vLLM 0.27.1 | none | `configs/nemotron-3-nano-30b-a3b/h200/config-21/` |
| Config 22 | H200 | FP8; DP8×TP1; FP8 KV cache; max-num-seqs 4; max-num-batched-tokens 16384; vLLM 0.27.1 | none | `configs/nemotron-3-nano-30b-a3b/h200/config-22/` |
| Config 23 | H200 | FP8; DP8×TP1; FP8 KV cache; max-num-seqs 256; max-num-batched-tokens 8192; vLLM 0.27.1 | none | `configs/nemotron-3-nano-30b-a3b/h200/config-23/` |

## Qwen3-32B

| Config | Instance | Configuration | Speculation | Folder |
|---|---|---|---|---|
| baseline | G7 | BF16; DP2×TP4; stock flags (only --max-model-len 12288 and no prefix caching); vLLM 0.27.1 | none | `configs/qwen3-32b/g7/baseline/` |
| Config 1 | G7 | NVFP4; DP4×TP2; FP8 KV cache; max-num-seqs 64; max-num-batched-tokens 8192; vLLM 0.27.1 | EAGLE-3 k=3, measured on this config (launch-eagle3.sh) | `configs/qwen3-32b/g7/config-1/` |
| Config 2 | G7 | NVFP4; DP4×TP2; FP8 KV cache; max-num-seqs 32; max-num-batched-tokens 2048; vLLM 0.27.1 | EAGLE-3 k=3, measured on this config (launch-eagle3.sh) | `configs/qwen3-32b/g7/config-2/` |
| Config 3 | G7 | NVFP4; DP4×TP2; FP8 KV cache; max-num-seqs 16; max-num-batched-tokens 4096; vLLM 0.27.1 | EAGLE-3 k=3, measured on this config (launch-eagle3.sh) | `configs/qwen3-32b/g7/config-3/` |
| Config 4 | G7 | NVFP4; DP4×TP2; FP8 KV cache; max-num-seqs 16; max-num-batched-tokens 2048; vLLM 0.27.1 | EAGLE-3 k=3, measured on this config (launch-eagle3.sh) | `configs/qwen3-32b/g7/config-4/` |
| Config 5 | G7 | NVFP4; DP4×TP2; FP8 KV cache; max-num-seqs 8; max-num-batched-tokens 2048; vLLM 0.27.1 | EAGLE-3 k=3, measured on this config (launch-eagle3.sh) | `configs/qwen3-32b/g7/config-5/` |
| Config 6 | G7 | NVFP4; DP2×TP4; FP8 KV cache; max-num-seqs 64; max-num-batched-tokens 4096; vLLM 0.27.1 | EAGLE-3 k=3, measured on this config (launch-eagle3.sh) | `configs/qwen3-32b/g7/config-6/` |
| Config 8 | G7 | NVFP4; DP4×TP2; FP8 KV cache; max-num-seqs 4; max-num-batched-tokens 2048; vLLM 0.27.1 | EAGLE-3 k=3, measured on this config (launch-eagle3.sh) | `configs/qwen3-32b/g7/config-8/` |
| Config 10 | G7 | NVFP4; DP4×TP2; FP8 KV cache; max-num-seqs 2; max-num-batched-tokens 2048; vLLM 0.27.1 | EAGLE-3 k=3, measured on this config (launch-eagle3.sh) | `configs/qwen3-32b/g7/config-10/` |
| baseline | G7e | BF16; one TP1 replica per GPU; stock flags (only --max-model-len 12288 and no prefix caching); vLLM 0.27.1 | none | `configs/qwen3-32b/g7e/baseline/` |
| Config 11 | G7e | NVFP4; DP8×TP1; FP8 KV cache; max-num-seqs 16; max-num-batched-tokens 16384; vLLM 0.27.1 | EAGLE-3 k=3, measured on this config (launch-eagle3.sh) | `configs/qwen3-32b/g7e/config-11/` |
| Config 17 | G7e | NVFP4; DP8×TP1; FP8 KV cache; max-num-seqs 64; max-num-batched-tokens 16384; vLLM 0.27.1 | EAGLE-3 k=3, measured on this config (launch-eagle3.sh) | `configs/qwen3-32b/g7e/config-17/` |
| Config 18 | G7e | NVFP4; DP8×TP1; FP8 KV cache; max-num-seqs 32; max-num-batched-tokens 16384; vLLM 0.27.1 | EAGLE-3 k=3, measured on this config (launch-eagle3.sh) | `configs/qwen3-32b/g7e/config-18/` |
| Config 19 | G7e | NVFP4; DP8×TP1; FP8 KV cache; max-num-seqs 8; max-num-batched-tokens 16384; vLLM 0.27.1 | EAGLE-3 k=3, measured on this config (launch-eagle3.sh) | `configs/qwen3-32b/g7e/config-19/` |
| baseline | H200 | BF16; one TP1 replica per GPU; stock flags (only --max-model-len 12288 and no prefix caching); vLLM 0.27.1 | none | `configs/qwen3-32b/h200/baseline/` |
| Config 12 | H200 | FP8; DP8×TP1; FP8 KV cache; max-num-seqs 64; max-num-batched-tokens 8192; vLLM 0.27.1 | EAGLE-3 k=3, measured on this config (launch-eagle3.sh) | `configs/qwen3-32b/h200/config-12/` |
| Config 13 | H200 | FP8; DP8×TP1; FP8 KV cache; max-num-seqs 256; max-num-batched-tokens 2048; vLLM 0.27.1 | EAGLE-3 k=3, measured on this config (launch-eagle3.sh) | `configs/qwen3-32b/h200/config-13/` |
| Config 14 | H200 | FP8; DP4×TP2; FP8 KV cache; max-num-seqs 64; max-num-batched-tokens 8192; vLLM 0.27.1 | EAGLE-3 k=3, measured on this config (launch-eagle3.sh) | `configs/qwen3-32b/h200/config-14/` |
| Config 15 | H200 | FP8; DP8×TP1; FP8 KV cache; max-num-seqs 16; max-num-batched-tokens 8192; vLLM 0.27.1 | EAGLE-3 k=3, measured on this config (launch-eagle3.sh) | `configs/qwen3-32b/h200/config-15/` |
| Config 16 | H200 | FP8; DP2×TP4; FP8 KV cache; max-num-seqs 256; max-num-batched-tokens 4096; FLASHINFER attention; vLLM 0.27.1 | none | `configs/qwen3-32b/h200/config-16/` |
| Config 20 | H200 | FP8; DP8×TP1; FP8 KV cache; max-num-seqs 128; max-num-batched-tokens 8192; vLLM 0.27.1 | EAGLE-3 k=3, borrowed from Config 12's arm (not re-measured here) | `configs/qwen3-32b/h200/config-20/` |
| Config 21 | H200 | FP8; DP8×TP1; FP8 KV cache; max-num-seqs 256; max-num-batched-tokens 8192; vLLM 0.27.1 | EAGLE-3 k=3, borrowed from Config 12's arm (not re-measured here) | `configs/qwen3-32b/h200/config-21/` |
| Config 22 | H200 | FP8; DP8×TP1; FP8 KV cache; max-num-seqs 8; max-num-batched-tokens 16384; vLLM 0.27.1 | EAGLE-3 k=3, borrowed from Config 12's arm (not re-measured here) | `configs/qwen3-32b/h200/config-22/` |
| Config 23 | H200 | FP8; DP8×TP1; FP8 KV cache; max-num-seqs 4; max-num-batched-tokens 16384; vLLM 0.27.1 | EAGLE-3 k=3, borrowed from Config 12's arm (not re-measured here) | `configs/qwen3-32b/h200/config-23/` |
| Config 24 | H200 | FP8; DP8×TP1; FP8 KV cache; max-num-seqs 64; max-num-batched-tokens 8192; FLASHINFER attention; vLLM 0.27.1 | EAGLE-3 k=3 measured, not applied: it loses under FlashInfer attention | `configs/qwen3-32b/h200/config-24/` |
| Config 25 | H200 | FP8; DP4×TP2; FP8 KV cache; max-num-seqs 64; max-num-batched-tokens 8192; FLASHINFER attention; vLLM 0.27.1 | EAGLE-3 k=3 measured, not applied: it loses under FlashInfer attention | `configs/qwen3-32b/h200/config-25/` |
