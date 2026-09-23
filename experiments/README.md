# experiments

实验目录。**一个主题一个目录**，每个目录都是完整可复现单元：入口 `run.sh` + 功能脚本 + 结果落盘位置。

```text
experiments/
├── README.md                          ← 本文件：横向索引与共同约定
├── vlm_inference_benchmark/                 ★ 主线研究：4GB 显存下的 VLM 推理（2B 模型）
│   ├── shared/                          跨引擎共用工具
│   └── vllm/                            vLLM 的实验（将来并列 sglang/ 等）
│       ├── exp1_memory_boundary/          显存可行边界扫描
│       ├── exp2_engine_comparison/        双栈对比（vLLM+AWQ vs HF+NF4）
│       ├── exp3_prefix_caching/           前缀缓存收益实测
│       └── preliminary_3b_memory_sweep/   前置实验（3B，已被取代）
└── bnb_kernel_align/                  纯微基准：bnb 4bit GEMM 的快慢路径
```

**按引擎分组**：`vlm_inference_benchmark/vllm/` 下全是 vLLM 的实验；
以后做 SGLang 就是在同一层加 `sglang/`，`shared/` 保持跨引擎共用。

**主线研究**请看 [`vlm_inference_benchmark/README.md`](vlm_inference_benchmark/README.md)，
它把各组实验的动机、逻辑关系、核心数据与坑都写在了一起。

> 📄 **完整报告**：[`../REPORT.md`](../REPORT.md)
> （自包含，含完整数据表、机制分析、排错过程与复现指引）

## 共同约定

1. **入口一律叫 `run.sh`**；功能脚本用功能名（`sweep.py` / `compare.py`），不带周次前缀。
2. **结果目录与实验目录同名同构**（`results/vlm_inference_benchmark/vllm/expN_xxx/`），
   由 `shared/common.py:results_dir_of()` 自动镜像 —— 重组分组时**不需要改任何路径**。
3. **分析脚本只打印、不落盘** —— 避免"脚本自动改文档"导致数据与叙述不一致。
4. **GPU 实验严格串行**：只有一块 4GB 卡，同时只允许一个推理进程。
5. **所有脚本用 venv 解释器**（`~/venvs/vllm/bin/python`）——torch 只装在 venv 里，
   系统 `python3` 没有。脚本内部通过 `shared/paths.sh` 的 `bench_python()` 自动处理。
6. **路径不写死层级**：用 `shared/paths.sh`（shell）或 `shared/common.py`（Python）
   向上找 `small-gpu-vlm-inference` 锚点。历史教训：脚本被移动后固定层级会**静默指错**。
7. **改完结构跑回归**：`bash tools/verify_refactor.sh`。

---

## vlm_inference_benchmark — 4GB 显存下的 VLM 推理研究

四个子目录：`shared/` + 三个实验。完整说明见该目录的 README，这里只放索引。

| 实验 | 问题 | 入口 |
|---|---|---|
| **exp1_memory_boundary** | 这台机器的显存可行边界在哪？ | `bash vlm_inference_benchmark/vllm/exp1_memory_boundary/run.sh` |
| **exp2_engine_comparison** | vLLM+AWQ 与 HF+NF4 差多少？ | `bash vlm_inference_benchmark/vllm/exp2_engine_comparison/run.sh` |
| **exp3_prefix_caching** | prefix caching 在共享前缀场景能省多少？ | `bash vlm_inference_benchmark/vllm/exp3_prefix_caching/run_frames.sh` |
| **sglang**（横向对比） | SGLang 的 radix 树在**同一份负载**上表现如何？ | `bash vlm_inference_benchmark/sglang/run_frames.sh` |

一键全跑 vLLM 三组：`bash vlm_inference_benchmark/run_all.sh`（约 45 分钟）

**核心数据**

| 实验 | 关键结论 |
|---|---|
| [exp1](vlm_inference_benchmark/vllm/exp1_memory_boundary/README.md) | `non_kv` **2610 MiB** · KV 池 0.67 GiB/25,008 tokens · KV 成本 **28 KiB/token** · gmu 可用区间 **0.65~0.80**（上限 0.802 是机器常数） |
| [exp2](vlm_inference_benchmark/vllm/exp2_engine_comparison/README.md) | **vLLM 在 batch 1/2/4/8 上快 3.5× / 3.0× / 2.6× / 2.5×**；代价是多用 1.71 GiB 显存、启动慢 4 倍 |
| [exp3](vlm_inference_benchmark/vllm/exp3_prefix_caching/README.md) | 缓存开启后延迟基本不随前缀增长（0.155→0.192 s），加速比 **2.43× → 12.76×**，命中率 99.1~99.8% |
| [sglang](vlm_inference_benchmark/sglang/README.md) | **同一份负载**移植完成（8 组零失败、视觉 token 差 0.08%）。SGLang 加速比 **1.78 / 1.06 / 1.37 / 1.70×**（非单调）。**⚠️ 两边 attention backend 不对等，绝对延迟不可比** —— 只比趋势与命中行为 |

每个实验目录下都有自己的 README：怎么跑、参数、输出文件、怎么读结果。

---

## vllm/preliminary_3b_memory_sweep — 3B 版边界扫描（前置，已被取代）

最初在 Qwen2.5-VL-3B 上做同一套扫描。3B 的 AWQ 权重 **3.17 GiB 装不下**（CUDA 可用仅 3.21 GiB），
vLLM 被迫 `cpu_offload_gb=1.0` → 解码走 PCIe → **性能差距无法归因到引擎**。

它留下了三条在 2B 上完全复现的方法论结论（三种死法签名 / gmu 上限 0.802 / mnbt 吃 KV 池），
所以保留作方法学记录。**模型已删除，不可复现**。

```bash
python experiments/vlm_inference_benchmark/vllm/preliminary_3b_memory_sweep/summarize.py   # 历史总表
```

详见 [`preliminary_3b_memory_sweep/README.md`](vlm_inference_benchmark/vllm/preliminary_3b_memory_sweep/README.md)。

---

## bnb_kernel_align — bnb 4bit GEMM 的快慢路径

**问题**：NF4 量化模型加载时报的 `inner dimension (3420) is not aligned for fast kernel with
blocksize=64, falling back to slower implementation` 到底意味着什么、慢多少？

> 完整的机制分析、实测数据（7 个配置）与四个发现都在
> **[`bnb_kernel_align/README.md`](bnb_kernel_align/README.md)**。
> 一句话结论：**这个规模下 4bit 比 bf16 更慢** —— 量化省的是显存，不是时间。

```bash
bash experiments/bnb_kernel_align/run.sh     # 约 2 分钟，不加载模型
```

**踩坑**：v1 靠「有没有打印警告」判断走了哪条路 —— **是错的**。源码里
`M > 1536` 与按架构校准的启发式都会**静默回退**（无警告）。必须用 profiler 看 kernel 名。