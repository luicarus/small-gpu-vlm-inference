# sglang/ — SGLang 引擎实验

与 [`../vllm/`](../vllm/) **并列**的引擎分组，用**同一份负载**对比两种前缀复用机制
（SGLang 的 radix 树 vs vLLM 的块哈希链）。

## 怎么跑

```bash
# 冒烟：先确认 SGLang 能在这台机器上跑起来（首次必做）
bash run_smoke.sh

# exp3 负载移植：帧数扫描（共享前缀长度）
bash run_frames.sh

# 单档调试
python run_cache.py --num-questions 10 --frames 16 --context-length 4096 \
  --mem-fraction-static 0.90 --out /tmp/test.jsonl
```

## 文件

| 文件 | 作用 |
|---|---|
| `smoke.py` | 冒烟：验证 2B AWQ 能加载、KV 池大小、prefix 命中是否工作 |
| `run_smoke.sh` | 冒烟入口（含环境变量设置） |
| `run_cache.py` | exp3 执行器：共享前缀负载，逐请求记录命中与墙钟 |
| `run_frames.sh` | 帧数扫描入口 |

## ⚠️ 三个环境前提（都是实测踩出来的）

### 1. 独立 venv

SGLang 要的 `transformers` 版本与 vLLM 不同（**5.12.1 vs 5.16.1**），
装进同一个 venv 会**静默改变依赖树**，让已跑通的三组 vLLM 实验结论失效。

```bash
python3 -m venv ~/venvs/sglang
~/venvs/sglang/bin/pip install "sglang[all]" -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

### 2. `CUDA_HOME` 必须指向 venv 内的 CUDA 13

本机系统 nvcc 是 **CUDA 12.0**，而 flashinfer 的 JIT 编译命令带
`--compress-mode=size`（CUDA 12.1+ 才有）→ 不指会被 `nvcc fatal: Unknown option` 拦住。
`run_smoke.sh` / `run_frames.sh` 已自动设置。

> 但这只解决"编译能过"；改用 CUDA 13.4 后
> **flashinfer 自带的 CCCL 头文件又会报版本不兼容** ——
> 所以最终仍要**避开 flashinfer**，见下条。

### 3. `--attention-backend triton`（绕开 flashinfer）

SGLang 默认走 flashinfer，而它在本机装不起来（见上）。
`triton` 后端不依赖 flashinfer，可用。

⚠️ **backend 选择对性能影响极大**（同一引擎、同一负载实测）：
`triton` 1.00× / `torch_native` 2.34× / `flex_attention` 4.01×。
所以**必须固定住**，且**与 vLLM 对比时不可比绝对延迟**
（vLLM 用 FA2，SGLang 只能用 triton）。

## 输出

`results/vlm_inference_benchmark/sglang/`

- `frames<K>_cache{on,off}.jsonl` —— 首行 meta，之后逐条请求
- `logs/<配置>.log`

## 怎么读结果

**主指标是 `wall_s`（本进程墙钟），统计时剔除以 0 条请求**（冷启动 + 一次性开销），
与 vLLM 版口径一致。

**命中数在 `meta_info["cached_tokens"]`**，语义与 vLLM 的 `num_cached_tokens` 一致
（都是"已驻留、会被复用的 KV"，见 `schedule_batch.py:2732`）。

**⚠️ 两个容易读错的地方**：

1. **`mem_fraction_static` 的语义与 vLLM 的 `gpu_memory_utilization` 相反**：
   它是"**有意留空（非 KV）的比例**"，调大 → slack 变小 → KV 变大。
   （源码：`kv_cache_configurator.py:2165`）
2. **长 context 需要更高的 `mem_fraction_static`**：
   长 context 抬高 profiling 的 activation 峰值 → 最小可用 mfs 变大。
   实测 ctx=6144 时 0.90 直接启动失败，需 0.92；ctx=8192 需 0.93。

## 与 vLLM 版的差异（口径对齐要点）

| 项 | vLLM | SGLang |
|---|---|---|
| 视频输入 | `{"video": [f1..fn]}` | **`video_data=[[f1..fn]]`**（多一层嵌套：按请求分组） |
| 像素预算 | `mm_processor_kwargs={"max_pixels": ...}` | **`mm_process_config={"video": {"max_pixels": ...}}`** |
| 命中数 | `RequestOutput.num_cached_tokens` | `out["meta_info"]["cached_tokens"]` |
| 生硬默认 | — | 视频每帧默认 **602,112** px（vLLM 侧我们统一 262,144） |

**不设 `mm_process_config` 的话，同一张图在两边会差 2.9× 的视觉 token** ——
比出来的是分辨率差异，不是引擎差异。

## 相关

- **机制对照（radix 树 vs 块哈希链）**：读双方源码的对比写在
  `REPORT.md` §4 与本目录 README 的「与 vLLM 版的差异」一节；
  核心差异是 vLLM 用**固定 16-token 块的哈希链**、SGLang 用**变长边的基数树**。
- **完整数据与可信度边界**：见 `results/vlm_inference_benchmark/sglang/` 的 8 份 JSONL，
  以及本文档「数据可信度边界」一节。
- vLLM 版同一实验：`../vllm/exp3_prefix_caching/`
