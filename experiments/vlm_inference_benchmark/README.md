# 4GB 显存下的 VLM 推理实验

在一张 **4GB 消费级显卡**（RTX 3050 Ti Laptop）上做的实验，按**引擎**分组
（`vllm/` 与 `sglang/` 并列，将来加别的引擎就是再加一层）。

## vLLM 侧：三组递进实验

| # | 问题 | 实验 | 结论（一句话） |
|---|---|---|---|
| 1 | 这台机器的**显存可行边界**在哪？ | [`vllm/exp1_memory_boundary/`](vllm/exp1_memory_boundary/) | 三种越界失败各有**不同的错误签名**，摸清后任何模型都能在几分钟内定位可用区间 |
| 2 | `vLLM+AWQ` 与 `Transformers+NF4` **差多少**？ | [`vllm/exp2_engine_comparison/`](vllm/exp2_engine_comparison/) | 关掉 offload 后 **vLLM 快 2.5~3.5×**，代价是多用 1.71 GiB 显存 |
| 3 | **prefix caching** 在共享前缀场景能省多少？ | [`vllm/exp3_prefix_caching/`](vllm/exp3_prefix_caching/) | 缓存开启后延迟基本不随前缀增长（0.155→0.192 s），加速比 **2.43× → 12.76×** |

三组实验**全部使用同一个模型** `Qwen2-VL-2B-Instruct`（AWQ 权重 2.74 GiB），
所以三份数据可以直接比较。

## SGLang 侧：横向对比

[`sglang/`](sglang/) 把 exp3 的**同一份共享前缀负载**移植过去，对比两种前缀复用机制
（SGLang 的 radix 树 vs vLLM 的块哈希链）。

实测加速比 **1.78 / 1.06 / 1.37 / 1.70×**（非单调，16 帧有凹陷）。

> ⚠️ **重要限制**：两边 attention backend **不对等** ——
> vLLM 用自带预编译的 FA2（`_vllm_fa2_C.so`），SGLang 在本机只能用 `triton`
> （其 flashinfer 路径与系统 CUDA 12.0 冲突）。
> 实测**同一引擎换 backend 就有 4 倍差距**，因此**绝对延迟不可横向比较**，
> 只能比趋势与命中行为。详见 [`sglang/README.md`](sglang/README.md)。

> 📄 **完整报告**：[`../../REPORT.md`](../../REPORT.md)（机制解释、适用边界、排错过程）
>
> 🔬 **每个实验目录下都有自己的 README**：怎么跑、参数、输出文件、怎么读结果。

---

## 为什么是「4GB + 2B VLM」这个组合

4GB 是一个**极端受限**的显存预算：2B 级 VLM 的 AWQ 权重就要 2.74 GiB，
留给 KV cache、激活值和 CUDA graph 的只剩 ~1.3 GiB。这种约束下：

- **参数不能乱设**——`gpu_memory_utilization` 高一点就启动失败，低一点 KV 池就空了；
- **引擎差异会被放大**——显存管理策略（预分配 vs 按需）直接决定"能不能跑起来"；
- **优化手段的有效性会变**——某些在数据中心有效的技术，在这里可能根本挤不出空间。

换句话说，**小显存是最好的显微镜**：它把平时被充裕资源掩盖掉的机制全部逼到台面上。

### 为什么用 2B 而不是更大的模型

最初在 3B 上做过一轮（见 [`../preliminary_3b_memory_sweep/`](../preliminary_3b_memory_sweep/)），
但发现 3B 在 4GB 上**无法消除混杂变量**：vLLM 被迫开启 `cpu_offload_gb`，
权重放在 CPU、解码时每个 token 都要走 PCIe（有效带宽 ~7 GB/s，比显存低一个数量级）。
这样一来测出的差距是"谁被迫 offload"的差距，**不能归因到引擎本身**。

换 2B 后权重降到 2.74 GiB，`offload=0` 装得下，**这笔"PCIe 税"消失**，剩下的才是引擎真实差距。

---

## 三组实验的逻辑关系

```
exp1 摸边界           exp2 比引擎            exp3 探优化
─────────────►        ─────────────►         ─────────────►
显存能怎么切？    →   同样预算下谁更快？  →   给定前缀复用能省多少？
gmu / 序列长 /         vLLM+AWQ vs            prefix caching
并发的可用区间         transformers+NF4       开/关对比
```

**exp1 是另外两个的地基**：它给出的 `non_kv`（权重+激活+graph 的固定占用）、
KV 池大小、KV/token 成本，正是解释 exp2/exp3 结果的量化语言。

---

## 核心数据

### exp1 · 显存账本（Qwen2-VL-2B-AWQ）

**gmu 可用区间：0.65 ~ 0.80**（14 配置实测，单调清晰）

| gmu | 预算 MiB | KV 池 | KV tokens | 并发上限 | 状态 |
|---|---|---|---|---|---|
| 0.55 / 0.60 | 2253 / 2458 | 0 | 0 | — | ❌ **#2 KV 池为空** |
| **0.65** | 2662 | 0.07 GiB | 2,544 | 2.48× | ✅ 下界 |
| 0.70 | 2867 | 0.27 | 10,032 | 9.80× | ✅ |
| 0.75 | 3072 | 0.47 | 17,520 | 17.11× | ✅ |
| **0.80** | 3277 | 0.67 | 25,008 | 24.42× | ✅ 上界 |
| 0.85 | 3482 | — | — | — | ❌ **#1 预算越界** |

| 量 | 实测值 | 说明 |
|---|---|---|
| **non_kv** | **2610 MiB** | 权重 + 峰值激活 + CUDA graph，与 gmu 无关 |
| **KV 成本** | **28.1 KiB/token** | 与架构推算一致：28 层 × 2(K/V) × 2(kv_heads) × 128(head_dim) × 2B = 28 KiB |
| **gmu 上限** | **0.802** | = CUDA 可用 3.21 GiB ÷ 总 4.0 GiB，是**机器常数**，与模型无关 |
| 每 +0.05 gmu | +0.20 GiB KV ≈ **+7,300 tokens** | 线性，可用于快速估算 |

**另外两组**（`max_model_len` / `max_num_seqs`）：序列越长，profiling 激活越大、KV 池越小
（len 1024→8192 时 non_kv 2610→3041 MiB，KV 池 0.67→0.25 GiB）；
并发 2/4/8 在 `max_num_batched_tokens=512` 下全部可行且 KV 池稳定在 0.68 GiB。

### exp2 · 双栈对比（batch 1/2/4/8）

| batch | HF + NF4 | vLLM + AWQ | vLLM 优势 |
|---|---|---|---|
| 1 | 73.46 s | **21.20 s** | **3.5×** |
| 8 | 28.61 s | **11.25 s** | 2.5× |

| | HF + NF4 | vLLM + AWQ |
|---|---|---|
| 吞吐（b8） | 35.4 tok/s | **86.9 tok/s** |
| 峰值显存 | **2249 MiB** | 3605 MiB |
| 启动时间 | **16 s** | 61 s |
| 峰值随 batch（b1→b8） | +416 MiB | **+26 MiB** |

### exp3 · prefix caching（共享前缀 = 伪视频）

| 帧数 | 前缀 token | 缓存关 | 缓存开 | 加速比 | 命中率 |
|---|---|---|---|---|---|
| 8 | 1,296 | 0.377 s | 0.155 s | 2.43× | 99.1% |
| 16 | 2,576 | 0.643 s | 0.170 s | 3.79× | 99.3% |
| 32 | 5,152 | 1.476 s | 0.171 s | 8.61× | 99.6% |
| 40 | 6,448 | 2.447 s | 0.192 s | **12.76×** | 99.8% |

**注意**：这一组数据在 2026-09-21 修正过一次。初版（1.66× → 3.10×、"残留 39 ms/帧"）
是在 GPU 异常状态下测得的，干净重测后被推翻。完整过程见 [`REPORT.md` §5.2](../../REPORT.md)。

---

## 目录结构

按**引擎**分组：`vllm/` 放全部 vLLM 实验，将来加 SGLang 就是并列的 `sglang/`。
`shared/` 是跨引擎共用工具，留在研究主题层。

```
vlm_inference_benchmark/
├── README.md                    本文件
├── run_all.sh                   一键依次跑三组实验
├── shared/                      跨引擎共用工具
│   ├── common.py                  路径锚点 / 结果目录镜像 / NVML 采样 / 提示词模板
│   └── paths.sh                   shell 侧锚点 + 基线守卫 + venv 解释器
└── vllm/                        ★ vLLM 的实验
    ├── exp1_memory_boundary/      显存边界扫描
    │   ├── sweep.py                 扫描编排（配置矩阵 + 子进程隔离 + 死法分类 + CUDA 门槛）
    │   ├── worker.py                单配置执行器
    │   ├── summarize.py             汇总成报告用表
    │   └── run.sh
    ├── exp2_engine_comparison/    双栈对比
    │   ├── run_engine.py            双引擎执行器
    │   ├── compare.py               四指标 + 跨引擎一致率分析
    │   ├── make_subset.py           OCRBench 子集生成
    │   └── run.sh / run_smoke.sh
    ├── exp3_prefix_caching/       前缀缓存实验
    │   ├── make_workload.py         共享前缀 workload 生成
    │   ├── run_cache.py             执行器（逐请求命中率 + 墙钟）
    │   ├── probe_frames.sh          帧数可行性探测
    │   └── run.sh / run_frames.sh / run_smoke.sh
    └── preliminary_3b_memory_sweep/  前置实验（3B，已被取代，不可复现）
```

**结果目录与实验目录同名同构**（由 `shared/common.py:results_dir_of()` 自动镜像）：

```
results/vlm_inference_benchmark/vllm/exp1_memory_boundary/
results/vlm_inference_benchmark/vllm/exp2_engine_comparison/
results/vlm_inference_benchmark/vllm/exp3_prefix_caching/
results/vlm_inference_benchmark/vllm/preliminary_3b_memory_sweep/
```

> 为什么要镜像：实验会随研究推进被**重新分组**（vLLM 的实验从顶层收进 `vllm/` 就是一次）。
> 手写结果路径的话，每次重组都要全文搜索改路径，**而且改漏了不会报错，只会写错地方**。
> 镜像规则让分组变化自动生效。

---

## 复现

### 前置

```bash
# 模型（走 hf-mirror；约 7 GiB）
bash setup/download_qwen2vl_2b.sh

# 环境自检：确认 venv / 显存 / 基线都正常
bash setup/gpu_check.sh
```

> ⚠️ 所有脚本必须用 venv 的解释器（`~/venvs/vllm/bin/python`）——
> torch 只装在 venv 里，**系统 `python3` 没有**。脚本内部已通过 `shared/paths.sh`
> 的 `bench_python()` 自动处理，无需手动指定。

### 跑实验

```bash
cd small-gpu-vlm-inference

# 一次全跑（约 45 分钟）
bash experiments/vlm_inference_benchmark/run_all.sh

# 或者分开跑
bash experiments/vlm_inference_benchmark/vllm/exp1_memory_boundary/run.sh
bash experiments/vlm_inference_benchmark/vllm/exp2_engine_comparison/run.sh
bash experiments/vlm_inference_benchmark/vllm/exp3_prefix_caching/run_frames.sh
```

> 前置实验（3B，已被取代、模型已删）的入口会明确提示不可复现，历史数据仍可汇总：
> `python experiments/vlm_inference_benchmark/vllm/preliminary_3b_memory_sweep/summarize.py`

### 环境要求

| 项 | 值 |
|---|---|
| GPU | RTX 3050 Ti Laptop **4GB**（sm86，20 SM） |
| 系统 | Windows + WSL2 (Ubuntu 24.04) |
| vLLM | 0.28.0（V1 引擎） |
| torch | 2.13.0+cu130 |
| transformers | 5.16.1（NF4 路径）· bitsandbytes 0.50.2 |

---

## 这台机器上的坑（都在代码注释里标了为什么）

跑通过程中撞到的问题，全部记在这里，因为它们会**静默地污染实验数据**：

| 现象 | 后果 | 处理 |
|---|---|---|
| **驱动回收滞后**：上个进程退出后 `nvidia-smi` 立刻显示已释放，**CUDA 侧却还没拿回空间** | vLLM 量到偏大的 `non_kv`（2610→3072 MiB）→ **本该成功的配置被判失败** | 等待条件用 **CUDA `mem_get_info()`** 而非 nvidia-smi（见 exp1 的 `wait_for_memory_release`） |
| **Triton 运行时 JIT**：vLLM 遇到 warmup 未覆盖的 shape 会在推理中途编译 kernel | 单批耗时 68 秒（占整轮 84%），只有看 P99 才发现 | 每个新 shape 首轮当 warmup 丢弃 |
| **WDDM 显存泄漏**：vLLM 初始化失败后进程退出但显存不回收 | 后续实验基线被抬高到 4000+ MiB，失败无法归因 | 基线守卫拦截；`Win+Ctrl+Shift+B` 重启驱动 |
| **`first_token_latency` 指标不可信**：两个时间戳来自不同来源 | 实测出现**负延迟** | 改用本进程 `perf_counter` 墙钟 |
| **`torch.cuda.max_memory_allocated()` 读不到** | vLLM 推理核心在子进程，父进程读到 ≈0 | 用 NVML 读整卡口径 |
| **`gpu_memory_utilization` 上限是常数 0.802** | 设 0.90 必然启动失败（死法 #1） | 上限 = CUDA 可用 ÷ 总显存 |
| **`max_num_batched_tokens` 会吃掉 KV 池** | 默认值时 batch≥2 直接死法 #2 | 显式压到 512 |

### 关于第一条的实测证据（它最隐蔽）

exp1 第一轮扫描出现**物理上不可能**的结果：gmu 0.65 成功、**0.70 失败**、0.75 失败、0.80 成功
（预算是单调递增的）。把 0.70 单独连跑 3 次 —— **3/3 全部成功，KV 恒 0.27 GiB**。

→ 失败是**假失败**：跑前基线用 nvidia-smi 看是干净的（21 MiB），但 CUDA 侧尚未完全恢复。
改用 CUDA 口径把门后，整个 A 组立刻变成完美的单调序列。

---

## 数据可信度

- 每组实验都**记录基线**，只在基线干净（<1000 MiB）时取数；
- 每个新 shape 的**首轮结果丢弃**（Triton JIT 一次性成本）；
- 延迟一律报**中位数与分位数**，不报均值（长尾会把均值带偏）；
- 失败配置**保留记录**并标注死法——边界的位置和成功案例同样重要；
- 不同引擎使用同一份提问集、同一解码参数（greedy）、同一像素预算。
