# vLLM serving benchmarks on AWS G7, G7e and H200

This repository runs the vLLM serving configurations that trace the throughput / latency / cost frontier of three models on three AWS instance types. Models:
Gemma-4-26B-A4B-IT, Qwen3-32B and Nemotron-3-Nano-30B-A3B. Three instances: G7
(g7.48xlarge, 8x RTX PRO 4500 32 GB), G7e (g7e.24xlarge, 4x RTX PRO 6000 96 GB) and H200
(p5en.48xlarge, 8x H200 SXM 141 GB). One workload everywhere: 160 chat requests with
coding prompts, input length p50 3,500 / p90 10,000 tokens, output length p50 200 / p90 400,
no prefix caching, temperature 0, sent as Poisson arrivals. Each deployment is measured at
a rising series of request rates until its throughput stops growing.

## Results

Improvement over stock vLLM (the BF16 checkpoint with default settings, one replica per GPU), as a multiplier: higher throughput and per-user speed, lower latency. Throughput is at its plateau. The latency and per-user columns depend on load, so they are given as a range from half of the baseline's capacity to 85 % of it. Absolute values for every configuration are in `configs/INDEX.md`.

| instance · model | throughput (output tok/s per GPU) | p99 TTFT (s) | p99 ITL (ms) | per-user speed (tok/s) |
|---|---:|---:|---:|---:|
| G7 · Gemma | 4.0× | 1.8–3.2× | 3.5–7.5× | 3.2–7.9× |
| G7 · Nemotron | 3.2× | 1.6–3.0× | 3.1–5.9× | 3.3–5.5× |
| G7 · Qwen | 3.3× | 2.1–2.5× | 4.2–9.8× | 4.1–8.1× |
| G7e · Gemma | 2.5× | 2.6–3.7× | 5.9–8.9× | 6.0–9.0× |
| G7e · Nemotron | 1.8× | 2.3–3.6× | 7.2–10.2× | 6.7–9.1× |
| G7e · Qwen | 3.4× | 2.4–6.4× | 4.8–7.0× | 5.8–9.1× |
| H200 · Gemma | 1.3× | 1.5–1.9× | 4.1–7.6× | 3.8–7.3× |
| H200 · Nemotron | 1.4× | 1.9–2.7× | 5.7–7.3× | 4.2–6.3× |
| H200 · Qwen | 1.6× | 4.2–5.7× | 5.1–7.5× | 4.9–5.7× |

`configs/INDEX.md` lists every configuration: precision, arrangement, KV cache, batch limits, flags, vLLM version and whether speculation was measured on it.

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
| `gemma-4-26B-A4B-it-MIXED-experts-W4A16-FP8` (ours) | `systalyze/gemma-4-26B-A4B-it-ExpertsINT4-FP8` | `eaeecbd06780820349102326a609fa0365005de5` |
| `Nemotron-3-Nano-30B-A3B-BF16` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | `bf77c3174f68ad409e1c2aa60daeb46e32d1c606` |
| `Nemotron-3-Nano-30B-A3B-FP8` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` | `9bee19446c0dfd01f356e10979d225b2a6621944` |
| `Nemotron-3-Nano-30B-A3B-NVFP4` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | `6efb4a2a1c1fa277ce7b3df7a1416255011b1c99` |
| `Qwen3-32B` | `Qwen/Qwen3-32B` | `9216db5781bf21249d130ec9da846c4624c16137` |
| `Qwen3-32B-FP8` | `Qwen/Qwen3-32B-FP8` | `aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df` |
| `Qwen3-32B-NVFP4` | `RedHatAI/Qwen3-32B-NVFP4` | `10a4cab9378ab938be865492b96ff42faff11c3f` |
| `Qwen3-32B-speculator.eagle3` (EAGLE-3 draft for Qwen) | `RedHatAI/Qwen3-32B-speculator.eagle3` | `dc84fe7ff1db31efa824776f49c141fc8195eb47` |
| `dspark-gemma4-26b-a4b` (ours, DSpark draft for Gemma-4) | `systalyze/gemma4-26b-a4b-dspark` | `4f186d61b54671bfe6a430ba0260a960ccf83da7` |

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
