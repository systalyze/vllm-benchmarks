# vLLM serving benchmarks on AWS G7, G7e and H200

This repository runs the vLLM serving configurations that trace the throughput / latency / cost frontier of three models on three AWS instance types. Models:
Gemma-4-26B-A4B-IT, Qwen3-32B and Nemotron-3-Nano-30B-A3B. Three instances: G7
(g7.48xlarge, 8x RTX PRO 4500 32 GB), G7e (g7e.24xlarge, 4x RTX PRO 6000 96 GB) and H200
(p5en.48xlarge, 8x H200 SXM 141 GB). One workload everywhere: 160 chat requests with
coding prompts, input length p50 3,500 / p90 10,000 tokens, output length p50 200 / p90 400,
no prefix caching, temperature 0, sent as Poisson arrivals. Each deployment is measured at
a rising series of request rates until its throughput stops growing.

## Results

Median inter-token latency of the best configuration, compared with the stock BF16 baseline
at the same load (85 % of the baseline's capacity).

| p50 ITL gain (x lower) | G7 - 8x RTX PRO 4500 32 GB (g7.48xlarge)† | G7e - 4x RTX PRO 6000 96 GB (g7e.24xlarge)* | H200 - 8x H200 SXM 141 GB (p5en.48xlarge) |
|---|---:|---:|---:|
| **Gemma-4-26B-A4B-IT** | **7.8x**§ | **9.1x** | **3.9x** |
| **Nemotron-3-Nano-30B-A3B** | **5.5x** | **5.2x** | **2.3x** |
| **Qwen3-32B** | **8.0x** | **9.0x**§ | **5.7x** |

- The baseline is the BF16 checkpoint on stock vLLM 0.27.1 with default flags, one replica
  per GPU, driven to its own throughput plateau.
- † On G7 the BF16 weights do not fit one or two 32 GB GPUs, so the baseline there is DP2xTP4.
- § The load point is below the lowest rate measured for that configuration, so the gain is
  a lower bound.
- \* AWS asked for the 8-GPU g7e.48xlarge, but none was obtainable in any region, so G7e was
  measured on the 4-GPU g7e.24xlarge. Every winner is one replica per GPU, so per-GPU numbers
  carry to the 8-GPU node. On G7 and H200 this was checked with full 8-GPU runs (within
  0.4 % and 0.03 %).

Load points, req/s per GPU: Gemma 0.729 / 3.488 / 7.640, Nemotron 0.971 / 5.040 / 9.956,
Qwen 0.165 / 0.529 / 1.290. Baseline capacities, req/s per GPU: 0.858 / 4.103 / 8.989,
1.142 / 5.929 / 11.713, 0.194 / 0.622 / 1.517.

Capacity, per-user speed and cost, baseline against the recommended deployment. Capacity is
output tokens/s per GPU at the plateau of the recommended configuration. Cost uses on-demand prices of $3.564 (G7), $4.1425 (G7e) and $7.912 (H200) per GPU-hour.

| instance · model | capacity, tok/s/GPU (baseline → recommended) | p50 ITL gain | per-user tok/s p50 (baseline → best) | $ per 1M output tokens (baseline → recommended) |
|---|---|---:|---|---|
| G7 · Gemma | 182 → 712 (3.9×) | 7.8×§ | 78 → 152 | $5.41 → $1.37 |
| G7 · Nemotron | 241 → 765 (3.2×) | 5.5× | 95 → 186 | $4.10 → $1.28 |
| G7 · Qwen | 38 → 129 (3.4×) | 8.0× | 31 → 99 | $25.23 → $7.68 |
| G7e · Gemma | 844 → 1,763 (2.1×) | 9.1× | 55 → 207 | $1.36 → $0.65 |
| G7e · Nemotron | 1,167 → 2,067 (1.8×) | 5.2× | 57 → 244 | $0.98 → $0.54 |
| G7e · Qwen | 136 → 409 (3.0×) | 9.0×§ | 16 → 88 | $8.44 → $2.49 |
| H200 · Gemma | 1,972 → 2,619 (1.3×) | 3.9× | 111 → 367 | $1.11 → $0.84 |
| H200 · Nemotron | 2,550 → 3,578 (1.4×) | 2.3× | 132 → 293 | $0.86 → $0.60 |
| H200 · Qwen | 343 → 503 (1.5×) | 5.7× | 42 → 182 | $6.26 → $4.35 |

`configs/INDEX.md` lists every configuration: precision, arrangement, KV
cache, batch limits, flags, vLLM version and whether speculation was measured on it.

## Running one configuration

You need:

- Docker with the NVIDIA container runtime, and the images listed at the end.
- Python 3.12. Install the client with
  `python3 -m venv bench/.venv && bench/.venv/bin/pip install -r bench/requirements.txt`
  (AIPerf 0.12.0; the driver refuses other versions).
- The checkpoints under `/data/models/<dir>`:
  `huggingface-cli download <repo> --revision <revision> --local-dir /data/models/<dir>`.
- Two empty cache directories some launch lines point at:
  `mkdir -p /data/fi-tactics /data/fi-tactics-lowmem`.

| `/data/models/<dir>` | HF repo | revision |
|---|---|---|
| `gemma-4-26B-A4B-it` | `google/gemma-4-26B-A4B-it` | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` |
| `gemma-4-26B-A4B-it-NVFP4` | `RedHatAI/gemma-4-26B-A4B-it-NVFP4` | `5557756b8dce33ac72f2bd702b11729fdba3b839` |
| `gemma-4-26B-A4B-it-FP8-dynamic` | `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` | `ed35d7abe5d940da41b4ff06eb482feb0be8cb44` |
| `gemma-4-26B-A4B-it-MIXED-experts-W4A16-FP8` (ours) | `systalyze/gemma-4-26B-A4B-it-MIXED-experts-W4A16-FP8` | `98bc35873f5f48f6a9f67f3f38358f0332dedcaf` |
| `Nemotron-3-Nano-30B-A3B-BF16` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | `bf77c3174f68ad409e1c2aa60daeb46e32d1c606` |
| `Nemotron-3-Nano-30B-A3B-FP8` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` | `9bee19446c0dfd01f356e10979d225b2a6621944` |
| `Nemotron-3-Nano-30B-A3B-NVFP4` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | `6efb4a2a1c1fa277ce7b3df7a1416255011b1c99` |
| `Qwen3-32B` | `Qwen/Qwen3-32B` | `9216db5781bf21249d130ec9da846c4624c16137` |
| `Qwen3-32B-FP8` | `Qwen/Qwen3-32B-FP8` | `aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df` |
| `Qwen3-32B-NVFP4` | `RedHatAI/Qwen3-32B-NVFP4` | `10a4cab9378ab938be865492b96ff42faff11c3f` |
| `Qwen3-32B-speculator.eagle3` (EAGLE-3 draft for Qwen) | `RedHatAI/Qwen3-32B-speculator.eagle3` | `dc84fe7ff1db31efa824776f49c141fc8195eb47` |
| `dspark-gemma4-26b-a4b` (ours, DSpark draft for Gemma-4) | `systalyze/dspark-gemma4-26b-a4b` | see the repo |

Two of the checkpoints are ours, published by Systalyze: the DSpark draft for Gemma-4 and the
mixed W4A16/FP8 Gemma-4 checkpoint. They download the same way as the others.

Then:

```
git clone --recurse-submodules <this repository>
cd vllm-benchmarks

bench/run.sh configs/gemma-4-26b-a4b-it/g7e/config-12 --dry-run     # print the plan, run nothing
bench/run.sh configs/gemma-4-26b-a4b-it/g7e/config-12               # run it
bench/run.sh configs/gemma-4-26b-a4b-it/g7e/config-12 --arm dspark  # same deployment with our draft
bench/run.sh configs/qwen3-32b/h200/config-14 --arm eagle3 --gpus 0,1  # use other GPUs than the launch line
```

What `bench/run.sh` does:

1. Reads the vLLM branch named at the top of the configuration's `launch.sh` and checks
   the `vllm/` submodule out at it.
2. Starts the containers and waits for `/health`.
3. Finds the saturation rate with a 3-minute closed-loop run.
4. Runs the workload at 0.3, 0.5, 0.7, 0.85 and 1.0 times that rate, then keeps stepping
   up by 0.1 until throughput stops growing. Deployments with `max-num-seqs` of 128 or
   more use 512 requests per rate instead of 160.
5. Stops the containers.

Results land in `runs/<name>-<UTC stamp>/`: one JSON per rate, the AIPerf output, the
container logs and a manifest. A speculation arm uses the exact-length workload variant
(`workload/aws-p50p90-v1-mintok`) automatically.

## What is where

- `configs/` - one folder per configuration on the frontier, plus a `baseline/` folder per model and instance. Each holds `launch.sh` (the exact `docker run` line) and `row.json` (the same deployment as data). Where speculation was measured there is also a `launch-dspark.sh` or `launch-eagle3.sh`. `INDEX.md` lists them all.
- `workload/` - the request lists, one per model, in two variants: `aws-p50p90-v1` for plain deployments, `aws-p50p90-v1-mintok` for speculation arms. `build_workload.py` generated them.
- `bench/` - `run.sh` and the driver behind it, `requirements.txt`, and the `Dockerfile` for the Gemma image.
- `vllm/` - vLLM as a git submodule. The launch lines mount patched files from it.

## The vLLM branches

The launch lines run stock vLLM images and bind-mount a few patched files from the `vllm/`
submodule over them, so nothing is rebuilt. Branch `systalyze/serving-0.27.1` is vLLM
v0.27.1 plus a small change that lets Gemma-4 use FlashInfer attention, and tuned kernel
configs for the H200 and RTX PRO 4500. Branch `systalyze/dspark-gemma4-0.28.0` is vLLM
v0.28.0 plus the fix that makes the Gemma-4 DSpark draft load, with the same FlashInfer
change. The submodule is checked out at the first branch. A configuration that needs the
second says `# vllm branch: systalyze/dspark-gemma4-0.28.0` at the top of its `launch.sh`,
and `bench/run.sh` switches to it before starting.

## Container images

- `vllm/vllm-openai:v0.27.1` - Qwen and Nemotron configurations on vLLM 0.27.1.
- `vllm/vllm-openai:v0.28.0` - the speculation arms and the Nemotron configurations marked vLLM 0.28.0.
- `systalyze/vllm-openai:v0.27.1-tf5.14.1` - the Gemma configurations on vLLM 0.27.1. It is the stock v0.27.1 image plus `transformers==5.14.1`, which 0.27.1 needs for Gemma-4. Build it with `docker build -f bench/Dockerfile -t systalyze/vllm-openai:v0.27.1-tf5.14.1 bench/`.
