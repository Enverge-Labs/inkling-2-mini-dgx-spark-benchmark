# Inkling 2 Mini on one DGX Spark

Serving speed of [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on a single DGX Spark (GB10), using vLLM's [DGX Spark recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini?hardware=dgx_spark_gb10) with the DSpark speculator. Run on 23 November 2026.

## Results

~1.2k tokens in, 1,024 out (reasoning included), temperature 1.0, client on the same machine.

| | 1 user | 16 users |
|---|---:|---:|
| SPEED-Bench `throughput_1k`, output tok/s | 49.6 | 198.1 |
| Time per output token, mean / median | 20.0 / 16.8 ms | 72.7 / 67.5 ms |
| Time to first token, median / P99 | 161 / 213 ms | 2.9 / 10.7 s |
| DSpark acceptance (tokens per step) | 28.9% (3.0) | 27.7% (2.9) |
| Random tokens 1k/1k, output tok/s | 36.4 | 174.6 |
| Random tokens, acceptance (tokens per step) | 16.6% (2.2) | 15.5% (2.1) |

## Setup

- DGX Spark, driver 580.142, CUDA 13.0
- `vllm/vllm-openai@sha256:7474ce29a0708258e5de643b44bac82755dadc4c3360ffff0ff6b2d0c53e19be` (vLLM `0.34.0rc1.dev1163`)

```bash
docker run -d --name inkling --gpus all --ipc=host -p 127.0.0.1:8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:nightly thinkingmachines/Inkling-2-Mini \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.5 \
  --reasoning-parser inkling \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

`--gpu-memory-utilization 0.5` is our change to the recipe. A GPU allocation that exhausts the Spark's unified memory can freeze the machine. At 0.5 the KV cache still holds 632,959 tokens, so the cap didn't limit any run here.

## Dataset

NVIDIA's SPEED-Bench [`prepare.py`](https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py) currently fails on `throughput_1k`, for two reasons:

- **StackSelect:** the dataset changed its `src_id` format, and the script raises `IndexError`.
- **HLE:** it's gated on Hugging Face.

`prepare_tolerant.py` runs NVIDIA's script but skips the rows it can't resolve. That keeps 1,024 of the 1,536 prompts: 512 low-entropy and 512 high-entropy.

```bash
curl -o speed_bench_prepare.py https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py
pip3 install --user datasets tiktoken requests aiohttp
python3 prepare_tolerant.py throughput_1k ./speed_bench
docker cp ./speed_bench inkling:/tmp/speed_bench
docker exec inkling pip install pandas "numpy==$(docker exec inkling python3 -c 'import numpy; print(numpy.__version__)')"
```

The nightly image lacks pandas, which the benchmark's SPEED-Bench loader needs. The `numpy==` pin keeps the image's installed numpy version.

## Benchmark

```bash
docker exec inkling vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
  --model thinkingmachines/Inkling-2-Mini \
  --dataset-name speed_bench --dataset-path /tmp/speed_bench \
  --speed-bench-dataset-subset throughput_1k --speed-bench-output-len 1024 \
  --max-concurrency 16 --num-prompts 64
```

- **1 user:** `--max-concurrency 1 --num-prompts 8`
- **Random tokens:** replace the dataset flags with `--dataset-name random --random-input-len 1024 --random-output-len 1024 --ignore-eos`

The recipe's own benchmark command leaves out `--dataset-path` and fails; see `results/bench_speed_recipe_as_written_c16.log`.

## Files

- `results/server.log`: the vLLM server from start to finish, covering download, load, KV-cache sizing, CUDA graphs and requests
- `results/bench_*.log`: one log per benchmark run
- `results/nvidia_smi.txt`, `results/vllm_image.txt`: the environment
