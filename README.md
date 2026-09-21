# small-gpu-vlm-inference

A reproducible **small-GPU VLM inference study** on a 4 GB RTX 3050 Ti, covering memory
boundaries, engine behaviour, and prefix caching — with the raw data, the failure modes,
and the measurement pitfalls all kept in the repo.

[中文说明见下](#中文说明)

---

## What's in here

Three experiments that build on each other:

| # | Question | Experiment |
|---|---|---|
| **1** | How much memory can actually be used, and what breaks when you exceed it? | [`exp1_memory_boundary`](experiments/vlm_inference_benchmark/vllm/exp1_memory_boundary/) |
| **2** | How do `vLLM + AWQ` and `Transformers + NF4` behave under the same hardware and workload, with CPU offload disabled? | [`exp2_engine_comparison`](experiments/vlm_inference_benchmark/vllm/exp2_engine_comparison/) |
| **3** | How much does prefix caching save when requests share a long prefix? | [`exp3_prefix_caching`](experiments/vlm_inference_benchmark/vllm/exp3_prefix_caching/) |

Plus a model-agnostic microbenchmark of bitsandbytes 4-bit GEMM
([`bnb_kernel_align`](experiments/bnb_kernel_align/)) and a preliminary study that was
superseded after we found a confound in it
([`preliminary_3b_memory_sweep`](experiments/vlm_inference_benchmark/vllm/preliminary_3b_memory_sweep/)).

**Full experiment write-up and analysis: [`REPORT.md`](REPORT.md)**

---

## Headline results

### 1. Memory boundaries are measurable, and there are three different failure modes

Model: `Qwen2-VL-2B-Instruct-AWQ`. Measured on an RTX 3050 Ti Laptop (4 GB, sm86, 20 SMs).

| Quantity | Measured | Note |
|---|---|---|
| `non_kv` (weights + peak activation + CUDA graph) | **2610 MiB** | independent of `gpu_memory_utilization` |
| KV cache cost | **28.1 KiB/token** | matches the architecture: 28 layers × 2 × 2 kv_heads × 128 × 2 B |
| Usable `gpu_memory_utilization` range | **0.65 – 0.80** | ceiling 0.802 = free CUDA memory ÷ total |

Exceeding the bounds gives three startup failures with **distinct error signatures** — mixing
them up just looks like "everything failed":

| Failure | Trigger | vLLM error |
|---|---|---|
| **#1 budget** | `gmu × total > free at startup` | `Free memory on device (3.21/4.0 GiB) ... is less than desired GPU memory utilization` |
| **#2 empty KV pool** | budget can't fit `non_kv` | `No available memory for the cache blocks` |
| **#3 sequence too long** | KV pool exists but can't hold one `max_model_len` sequence | `To serve at least one request with the model's max seq len ...` |

Failure #3 even reports the ceiling for you (`the estimated maximum model length is 7440`).

### 2. With offload disabled, vLLM was 2.5–3.5× faster than Transformers + bitsandbytes

100 OCRBench items, identical inputs and decoding parameters, 800 generations, zero failures,
`offload=0` in all runs. **This is an engineering comparison of two deployable stacks, not an
isolated engine-only comparison** — the quantization formats differ (AWQ vs NF4) and the vision
tower is quantized on one side only. See [REPORT.md §3.4](REPORT.md) for what this can and
cannot show.

| batch | Transformers + NF4 | vLLM + AWQ | speedup | HF peak | vLLM peak |
|---|---|---|---|---|---|
| 1 | 73.46 s | **21.20 s** | **3.5×** | **1833 MiB** | 3579 MiB |
| 2 | 52.07 s | **17.60 s** | 3.0× | **1899 MiB** | 3579 MiB |
| 4 | 34.38 s | **13.29 s** | 2.6× | **2065 MiB** | 3579 MiB |
| 8 | 28.61 s | **11.25 s** | 2.5× | **2249 MiB** | 3605 MiB |

Two differences fell out of the batch sweep:

- **Memory behaviour differs.** From batch 1 to 8, the HF stack's peak grows by **+416 MiB**
  while vLLM's grows by **+26 MiB** — PagedAttention pre-allocates the KV pool at startup and
  then only consumes it.
- **At batch 8 the HF stack's per-request latency gets worse** (P50 0.269 → 0.353 s) while vLLM
  keeps improving (0.102 → 0.088 s). Static batching pads to the longest sequence and makes the
  whole batch wait; iteration-level scheduling does not.

**Before comparing engines, first check whether either stack is offloading weights.**
In this 4 GB setup, CPU offload was large enough to dominate the engine-level differences —
a preliminary run on a larger model showed vLLM *losing* 1.8× under exactly those conditions.

### 3. Prefix caching keeps cached-path latency nearly flat as the shared prefix grows

Workload: one image repeated into K frames (a pseudo-video) as a shared prefix, plus N distinct
questions. The baseline disables the feature explicitly — **vLLM enables prefix caching by
default**, so "on vs on" would measure nothing.

| frames | prefix tokens | caching off | caching on | speedup | hit rate |
|---|---|---|---|---|---|
| 8 | 1,296 | 0.377 s | 0.155 s | **2.43×** | 99.1% |
| 16 | 2,576 | 0.643 s | 0.170 s | **3.79×** | 99.3% |
| 32 | 5,152 | 1.476 s | 0.171 s | **8.61×** | 99.6% |
| 40 | 6,448 | 2.447 s | 0.192 s | **12.76×** | 99.8% |

Cached-path latency stays in a narrow band (0.155 → 0.192 s) while the prefix grows 5×, whereas
the uncached path grows 6.5×. The uncached cost grows super-linearly over the tested range,
**consistent with** the increasing attention cost of longer prefills.

With N distinct questions against a fixed prefix, the per-request speedup stays roughly constant
(~3.6–4.6×) while the whole-run speedup grows (1.66× → 3.17×) as engine fixed costs amortize.
So the gain comes from the prefix being large, not from asking many questions.

---

## Measurement pitfalls

These are cases where the instrument itself was wrong, and the wrong answer looked normal.
Details in [REPORT.md §5](REPORT.md).

| Pitfall | What it looks like | How to catch it |
|---|---|---|
| **Triton runtime JIT** | one batch takes 68 s (84% of the run); P50 is perfectly normal | report percentiles; discard the first run per shape |
| **`first_token_latency` gave negative values** | impossible numbers from a framework metric | cross-check the metric against wall-clock timing before using it |
| **"GPU ghost state"** | NVML shows 677 MiB used at 100% util with no process anywhere, while CUDA reports the device empty; configs that should succeed fail, and timings go bimodal | check **both** NVML and CUDA `mem_get_info()` — each one alone misses a case |
| **Contaminated data tells a plausible story** | a clean-looking trend (a constant "39 ms/frame") that turned out to be the artifact's fixed overhead | verify the environment between every step; keep discarded data in the repo |
| **WDDM leak after a failed engine init** | hundreds of MiB to 3+ GiB held with no process owning it; `wsl --shutdown` doesn't help | baseline guard; on Windows, restart the graphics driver |
| **A warning that isn't the bottleneck** | bnb warns about inner-dimension alignment, but the real cost is cuBLAS kernel selection | never infer the execution path from whether a warning printed |

---

## Layout

```
small-gpu-vlm-inference/
├── REPORT.md              full experiment write-up and analysis
├── setup/                 environment install + sanity checks
├── engines/               minimal engine smoke tests (vLLM / transformers) + HTTP serving demo
├── experiments/
│   ├── vlm_inference_benchmark/
│   │   ├── shared/          path anchors, NVML sampling, prompt templates
│   │   └── vllm/            grouped by engine
│   │       ├── exp1_memory_boundary/
│   │       ├── exp2_engine_comparison/
│   │       ├── exp3_prefix_caching/
│   │       └── preliminary_3b_memory_sweep/
│   └── bnb_kernel_align/  bitsandbytes 4-bit GEMM microbenchmark
├── results/               raw JSONL + logs (evidence for every number in the report)
├── assets/                test image (see its README before replacing it)
└── tools/                 result viewer + structure regression check
```

`results/` mirrors `experiments/` one-to-one — the mapping is computed, not hard-coded,
so moving an experiment around can't silently redirect its output.

---

## Reproduce

```bash
# 1. environment
bash setup/install_vllm.sh
bash setup/install_bitsandbytes.sh
bash setup/install_accelerate.sh
bash setup/download_qwen2vl_2b.sh
bash setup/gpu_check.sh

# 2. all three experiments, serially (~45 min)
bash experiments/vlm_inference_benchmark/run_all.sh

# 3. or individually
bash experiments/vlm_inference_benchmark/vllm/exp1_memory_boundary/run.sh
bash experiments/vlm_inference_benchmark/vllm/exp2_engine_comparison/run.sh
bash experiments/vlm_inference_benchmark/vllm/exp3_prefix_caching/run_frames.sh
```

### Environment used

| | |
|---|---|
| GPU | RTX 3050 Ti Laptop, 4 GB, sm86, 20 SMs |
| OS | Windows + WSL2 (Ubuntu 24.04) |
| vLLM / torch | 0.28.0 (V1 engine) / 2.13.0+cu130 |
| transformers / bitsandbytes | 5.16.1 / 0.50.2 |
| Model | `Qwen2-VL-2B-Instruct` (bf16 and AWQ builds) |

Python dependencies live in a venv (`~/venvs/vllm`), never in the system interpreter — vLLM
pins specific torch/CUDA builds and Ubuntu 24.04 blocks system-wide `pip install` (PEP 668).
Every script resolves the interpreter through `shared/paths.sh`.

### Data handling

Each result file records the baseline before the run, and runs are accepted only when both
NVML and CUDA report a clean device. Failed configurations are kept with their failure mode
labelled — where the boundary sits matters as much as the successes. Latency is reported as
median and percentiles, since long tails are real and they skew means.

---

## 中文说明

本仓库是在一张 4GB 显卡上做的 **VLM 推理实验**，包含三组递进的实验：

1. **显存可行边界**——测出 `gpu_memory_utilization`、序列长度、并发度的可用区间，
   并区分三种错误签名各不相同的启动失败。
2. **引擎对比**——在**关闭 CPU offload** 的前提下比较 `vLLM + AWQ` 与 `Transformers + NF4`。
   注意这是**两套可部署方案的工程对比，不是纯引擎对比**：量化格式不同（AWQ vs NF4），
   且只有一侧量化了视觉塔。结论的适用边界写在报告 §3.4。
3. **前缀缓存收益**——共享前缀从 1,296 增到 6,448 token 时加速比 **2.43× → 12.76×**，
   而缓存开启后的延迟基本不变（0.155 → 0.192 s），命中率 99.1~99.8%。

此外还有 bnb 4-bit GEMM 的微基准，以及一组**被取代的前置实验**。

**报告里最值得看的是第 5 节**：实验过程中有两次测量方法本身出了问题——
框架指标给出负延迟；GPU 进入异常状态污染了结果并导出一个错误结论。
那一节记录了怎么发现、怎么定位，以及重测后推翻了什么。

**完整报告见 [`REPORT.md`](REPORT.md)**；原始数据在 `results/`，每个数字都能追溯。

---

## License

MIT (code and documentation; see `assets/README.md` regarding the test image)
