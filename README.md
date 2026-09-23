# small-gpu-vlm-inference

A reproducible VLM inference study on a **4 GB RTX 3050 Ti Laptop GPU**, covering memory
limits, inference stacks, prefix caching, and a SGLang RadixAttention comparison.

Model: `Qwen2-VL-2B-Instruct` · WSL2 / Ubuntu 24.04 · vLLM 0.28.0 · SGLang 0.5.20

[中文说明](#中文说明) · [Full report](REPORT.md)

---

## Experiments

| # | Experiment | Question |
|---|---|---|
| 1 | [`exp1_memory_boundary`](experiments/vlm_inference_benchmark/vllm/exp1_memory_boundary/) | How much GPU memory can vLLM actually use on a 4 GB card? |
| 2 | [`exp2_engine_comparison`](experiments/vlm_inference_benchmark/vllm/exp2_engine_comparison/) | How do `vLLM + AWQ` and `Transformers + NF4` behave with CPU offload disabled? |
| 3 | [`exp3_prefix_caching`](experiments/vlm_inference_benchmark/vllm/exp3_prefix_caching/) | How much does prefix caching save as the shared prefix grows? |
| 4 | [`sglang`](experiments/vlm_inference_benchmark/sglang/) | How does SGLang's RadixAttention differ from vLLM's block-based prefix cache? |

A preliminary 3B experiment and a bitsandbytes 4-bit GEMM microbenchmark are also kept in
the repository, but are not part of the main benchmark.

---

## Results

### 1. vLLM memory boundary

For `Qwen2-VL-2B-Instruct-AWQ`:

| Quantity | Measured |
|---|---:|
| `non_kv` at `max_model_len=1024` | **2610 MiB** |
| KV cache cost | **28.1 KiB/token** |
| usable `gpu_memory_utilization` | **0.65–0.80** |
| KV capacity at `gmu=0.80` | **25,008 tokens** |

The usable KV budget is approximately:

```text
KV memory ≈ gpu_memory_utilization × total VRAM − non_kv
KV tokens ≈ KV memory / 28.1 KiB
```

Three different startup failures were observed:

- requested memory budget exceeds free GPU memory;
- the remaining budget cannot allocate any KV blocks;
- the KV pool exists but cannot hold one `max_model_len` sequence.

Increasing `max_model_len` also increases profiling-time activation memory, reducing the
space left for KV cache.

### 2. vLLM + AWQ vs Transformers + NF4

100 OCRBench samples, identical inputs and greedy decoding, batch sizes 1/2/4/8, with
`cpu_offload_gb=0`.

| batch | Transformers + NF4 | vLLM + AWQ | speedup | HF peak | vLLM peak |
|---|---:|---:|---:|---:|---:|
| 1 | 73.46 s | **21.20 s** | **3.5×** | 1833 MiB | 3579 MiB |
| 2 | 52.07 s | **17.60 s** | 3.0× | 1899 MiB | 3579 MiB |
| 4 | 34.38 s | **13.29 s** | 2.6× | 2065 MiB | 3579 MiB |
| 8 | 28.61 s | **11.25 s** | 2.5× | 2249 MiB | 3605 MiB |

vLLM is faster across all tested batch sizes, while using substantially more GPU memory at
batch 1. Its peak allocation remains nearly flat as batch grows:

```text
Transformers: 1833 → 2249 MiB  (+416 MiB)
vLLM:        3579 → 3605 MiB   (+26 MiB)
```

This is a comparison of two deployable stacks, **not a pure engine-only comparison**: the
quantization formats and quantization coverage differ.

CPU offload must also be controlled. In the preliminary 3B experiment, enabling offload was
enough to reverse the performance result because decoding became limited by CPU↔GPU weight
transfers.

### 3. Prefix caching

The workload repeats one image into K frames to form a shared multimodal prefix, followed by
different questions.

| frames | prefix tokens | cache off | cache on | speedup | hit rate |
|---:|---:|---:|---:|---:|---:|
| 8 | 1,296 | 0.377 s | 0.155 s | **2.43×** | 99.1% |
| 16 | 2,576 | 0.643 s | 0.170 s | **3.79×** | 99.3% |
| 32 | 5,152 | 1.476 s | 0.171 s | **8.61×** | 99.6% |
| 40 | 6,448 | 2.447 s | 0.192 s | **12.76×** | 99.8% |

The uncached path grows from 0.377 s to 2.447 s, while cached-path latency stays between
0.155 s and 0.192 s.

The experiment is serial (`max_num_seqs=1`); cache behaviour under concurrent pressure is not
covered.

### 4. SGLang RadixAttention

vLLM and SGLang solve the same prefix-reuse problem with different data structures:

| | vLLM 0.28 | SGLang 0.5.20 |
|---|---|---|
| index | block hash + hash table | radix tree |
| prefix representation | KV blocks | variable-length token paths |
| context relation | parent block hash | tree path |
| tested match granularity | 16 tokens | token-level with `page_size=1` |

The core operations are:

```python
# vLLM
hash(parent_block_hash, block_tokens, extra_keys)

# SGLang
prefix_len = child.key.match(key, page_size=self.page_size)
if prefix_len < len(child.key):
    self._split_node(...)
```

After aligning the multimodal preprocessing budget, the two engines produce almost identical
prompt lengths:

```text
vLLM:   1307 prompt tokens / 1296 cached
SGLang: 1306 prompt tokens / 1294 cached
```

SGLang also reaches ~99% cache hit rates, but **absolute latency is not directly comparable**
on this machine:

```text
vLLM   → FlashAttention 2
SGLang → Triton
```

SGLang's default FlashInfer path could not be used with the system CUDA toolchain. The
comparison therefore focuses on cache behaviour and implementation differences, not an engine
speed ranking.

---

## Measurement notes

Two issues materially changed the final results.

First, vLLM's `first_token_latency` produced negative values in this setup, so latency
measurements were replaced with process-local `perf_counter()` wall-clock timing.

Second, failed engine initialization could leave the GPU in an inconsistent state where NVML
reported memory/utilization while CUDA reported the device as free. This contaminated later
measurements. Runs are now accepted only when both NVML and CUDA report a clean baseline.

Discarded runs are kept under `results/` for traceability.

See [REPORT.md](REPORT.md) for the full investigation.

---

## Reproduce

### vLLM / Transformers

```bash
bash setup/install_vllm.sh
bash setup/install_bitsandbytes.sh
bash setup/install_accelerate.sh
bash setup/download_qwen2vl_2b.sh
bash setup/gpu_check.sh

bash experiments/vlm_inference_benchmark/run_all.sh
```

Individual experiments:

```bash
bash experiments/vlm_inference_benchmark/vllm/exp1_memory_boundary/run.sh
bash experiments/vlm_inference_benchmark/vllm/exp2_engine_comparison/run.sh
bash experiments/vlm_inference_benchmark/vllm/exp3_prefix_caching/run_frames.sh
```

### SGLang

SGLang uses a separate virtual environment because its dependency versions differ from the
vLLM environment.

```bash
python3 -m venv ~/venvs/sglang
~/venvs/sglang/bin/pip install "sglang[all]"

export CUDA_HOME=~/venvs/sglang/lib/python3.12/site-packages/nvidia/cu13

bash experiments/vlm_inference_benchmark/sglang/run_smoke.sh
bash experiments/vlm_inference_benchmark/sglang/run_frames.sh
```

---

## Environment

| | |
|---|---|
| GPU | RTX 3050 Ti Laptop, 4 GB, sm86 |
| OS | Windows + WSL2 / Ubuntu 24.04 |
| Model | Qwen2-VL-2B-Instruct |
| vLLM | 0.28.0 |
| SGLang | 0.5.20 |
| PyTorch | 2.13.0+cu130 |
| Transformers | 5.16.1 |
| bitsandbytes | 0.50.2 |

---

## Repository layout

```text
small-gpu-vlm-inference/
├── REPORT.md
├── setup/
├── engines/
├── experiments/
│   ├── vlm_inference_benchmark/
│   │   ├── shared/
│   │   ├── vllm/
│   │   │   ├── exp1_memory_boundary/
│   │   │   ├── exp2_engine_comparison/
│   │   │   ├── exp3_prefix_caching/
│   │   │   └── preliminary_3b_memory_sweep/
│   │   └── sglang/
│   └── bnb_kernel_align/
├── results/
├── assets/
└── tools/
```

`results/` mirrors the experiment layout and contains raw JSONL/log files for the reported
measurements.

---

## 中文说明

这个仓库记录了 RTX 3050 Ti 4GB 上的几组 VLM 推理实验：

- vLLM 的显存边界和 KV Cache 容量；
- `vLLM + AWQ` 与 `Transformers + NF4` 的速度、显存和 batch 行为；
- 长共享前缀下 Prefix Caching 的实际收益；
- vLLM Block Hash 与 SGLang RadixAttention 的实现和命中行为差异。

核心结果：

```text
vLLM 可用 gmu：   0.65–0.80
KV Cache：        28.1 KiB/token
vLLM vs HF：      2.5–3.5×
Prefix Cache：    2.43× → 12.76×
```

SGLang 横向实验已经对齐视觉 token 数和缓存命中口径，但由于本机两边使用的 attention
backend 不同，不比较绝对延迟。

完整实验设计、限制和异常数据排查见 [`REPORT.md`](REPORT.md)。

---

## License

MIT

See `assets/README.md` for the test image license/source information.
