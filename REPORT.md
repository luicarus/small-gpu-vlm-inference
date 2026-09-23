# 4GB 显存下的 VLM 推理实验

模型：`Qwen2-VL-2B-Instruct`（AWQ / bf16）
硬件：RTX 3050 Ti Laptop 4GB，WSL2 / Ubuntu 24.04
软件：vLLM 0.28.0、SGLang 0.5.20、transformers 5.16.1、bitsandbytes 0.50.2、torch 2.13.0+cu130

本文记录四组实验：

1. vLLM 在 4GB 显存下的可用参数边界；
2. `vLLM + AWQ` 与 `Transformers + NF4` 的推理差异；
3. vLLM Prefix Caching 在长共享前缀下的收益；
4. SGLang RadixAttention 在相同负载下的横向实验。

主要结果：

- vLLM 的 `gpu_memory_utilization` 在本机可用区间为 **0.65～0.80**；
- `offload=0` 时，vLLM 在 batch 1/2/4/8 上比 Transformers + NF4 快 **2.5～3.5×**；
- 共享前缀从 1,296 增长到 6,448 token 时，vLLM Prefix Caching 的单请求加速从 **2.43×** 增至 **12.76×**；
- SGLang 能获得相近的前缀命中率，但本机只能使用 Triton attention backend，而 vLLM 使用 FA2，因此两边的绝对延迟不能直接比较。

---

## 1. 实验条件

最初使用过 Qwen2.5-VL-3B-AWQ，但其权重约 3.17 GiB，而本机 CUDA 启动时实际可用显存约 3.21 GiB。模型必须开启 `cpu_offload_gb` 才能运行，此时解码过程会持续从 CPU 搬运权重，性能主要受 PCIe 带宽影响。

因此正式实验换为 Qwen2-VL-2B-Instruct。AWQ 权重约 2.74 GiB，可以在 `offload=0` 下运行。3B 的实验代码和数据保留在 `preliminary_3b_memory_sweep/`，不参与后续结论。

测量统一采用以下口径：

| 指标 | 口径 |
|---|---|
| GPU 显存 | NVML 整卡采样，100 ms 间隔 |
| 延迟 | 中位数和 P99 |
| 解码 | `temperature=0`，固定 `max_tokens` |
| Warmup | 新 shape 的第一轮不计入结果 |
| GPU 基线 | 每个配置运行前检查 NVML 和 CUDA 两套显存状态 |

vLLM 0.28 的 EngineCore 在独立子进程中运行，因此父进程的 `torch.cuda.max_memory_allocated()` 不能代表整套引擎显存，本实验统一使用 NVML。

---

## 2. vLLM 显存边界

首先扫描 `gpu_memory_utilization`（下文简称 gmu）。

| gmu | 预算 MiB | KV 池 | KV tokens | 状态 |
|---|---:|---:|---:|---|
| 0.55 | 2253 | 0 | 0 | 失败 |
| 0.60 | 2458 | 0 | 0 | 失败 |
| **0.65** | 2662 | 0.07 GiB | 2,544 | 成功 |
| 0.70 | 2867 | 0.27 GiB | 10,032 | 成功 |
| 0.75 | 3072 | 0.47 GiB | 17,520 | 成功 |
| **0.80** | 3277 | 0.67 GiB | 25,008 | 成功 |
| 0.85 | 3482 | — | — | 失败 |

本机可用区间为 **0.65～0.80**。gmu 每增加 0.05，KV 池大约增加 0.20 GiB。

启动失败主要有三种：

| 类型 | 条件 |
|---|---|
| 显存预算超过启动时空闲显存 | `gmu × total_memory > free_memory` |
| KV 池为空 | 显存预算不足以容纳模型和非 KV 开销 |
| 最大序列过长 | KV 池存在，但放不下一条 `max_model_len` 请求 |

第三种情况下，vLLM 会直接打印当前显存条件下可支持的最大序列长度。例如本机曾返回：

```text
the estimated maximum model length is 7440
```

`max_model_len` 对 KV 池也有明显影响：

| 配置 | non_kv MiB | KV 池 | KV tokens |
|---|---:|---:|---:|
| len 1024 | 2610 | 0.67 GiB | 25,008 |
| len 2048 | 2672 | 0.61 GiB | 22,864 |
| len 4096 | 2795 | 0.49 GiB | 18,336 |
| len 8192 | 3041 | 0.25 GiB | 9,184 |
| seqs 2，mnbt 512 | 2580 | 0.70 GiB | 26,048 |
| seqs 4，mnbt 512 | 2611 | 0.68 GiB | 25,312 |
| seqs 8，mnbt 512 | 2611 | 0.68 GiB | 25,328 |

`max_model_len` 增大时，profiling forward 的激活峰值随之增加，因此 `non_kv` 上升，留给 KV Cache 的空间减少。

相反，在显式设置：

```text
max_num_batched_tokens = 512
```

以后，`max_num_seqs=2/4/8` 的 KV 池基本保持在 0.68～0.70 GiB。默认配置下 batch≥2 会因为非 KV 开销过高导致 KV 池为空。

本机在 `max_model_len=1024` 下的 `non_kv` 约为 **2610 MiB**。KV Cache 理论成本为：

```text
28 layers × 2(K/V) × 2 kv_heads × 128 head_dim × 2 bytes
≈ 28 KiB/token
```

实测约 **28.1 KiB/token**。

因此可以用下面的关系估算：

```text
KV 可用空间 ≈ gmu × 总显存 − non_kv

KV token 数 ≈ KV 可用空间 / 28.1 KiB
```

---

## 3. vLLM 与 Transformers

使用 OCRBench 100 条固定子集，两个引擎采用相同输入、贪心解码和像素预算 `max_pixels=262144`。batch 分别为 1、2、4、8，共执行 800 次推理。

所有 vLLM 配置均为 `offload=0`。

| batch | 引擎 | 总耗时 s | 吞吐 tok/s | 峰值 MiB | P50 s | P99 s | 加载 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | HF + NF4 | 73.46 | 13.7 | **1833** | 0.702 | 1.774 | **16.0** |
| 1 | vLLM + AWQ | **21.20** | **46.1** | 3579 | **0.173** | **0.474** | 73.0 |
| 2 | HF + NF4 | 52.07 | 19.3 | **1899** | 0.459 | 1.118 | **12.9** |
| 2 | vLLM + AWQ | **17.60** | **55.6** | 3579 | **0.146** | **0.350** | 63.8 |
| 4 | HF + NF4 | 34.38 | 29.4 | **2065** | 0.269 | 0.613 | **15.4** |
| 4 | vLLM + AWQ | **13.29** | **73.6** | 3579 | **0.102** | **0.279** | 63.7 |
| 8 | HF + NF4 | 28.61 | 35.4 | **2249** | 0.353 | 0.518 | **16.4** |
| 8 | vLLM + AWQ | **11.25** | **86.9** | 3605 | **0.088** | **0.219** | 61.2 |

vLLM 在四个 batch 下都更快，加速比由 batch=1 的 **3.5×** 降到 batch=8 的 **2.5×**。

显存行为则相反：

```text
HF：   1833 → 2249 MiB，+416 MiB
vLLM：3579 → 3605 MiB，+26 MiB
```

HF 的显存随 batch 增长；vLLM 启动时已经预留 KV Cache，因此 batch 增长后峰值基本不变。

batch=8 时 HF 的 P50 从 0.269 s 回升至 0.353 s，而 vLLM 从 0.102 s 继续下降到 0.088 s。这与两边批处理方式不同有关：HF 的静态 batch 需要处理 padding，并等待整批完成；vLLM 使用迭代级调度。

准确率只用于检查输出是否发生明显退化。batch=1 时两边均为 85%，batch=4 时 HF 为 84%、vLLM 为 85%。

这组实验不是严格的"只比较引擎"。vLLM 使用 AWQ，Transformers 使用 NF4，而且视觉模块的量化范围并不完全一致。因此可以比较当前两套实际部署栈的速度和显存行为，但不能把差异全部归因于 vLLM 和 Transformers 本身。

---

## 4. Prefix Caching

实验负载使用同一张图片重复为多帧，并给同一个视觉前缀配不同问题。这样可以控制共享前缀长度。

vLLM 0.28 默认开启 Prefix Caching，因此关闭组显式使用：

```text
--no-enable-prefix-caching
```

固定问题数 N=10，只改变前缀长度：

| 帧数 | 前缀 token | Cache Off | Cache On | 加速比 | 命中率 |
|---:|---:|---:|---:|---:|---:|
| 8 | 1,296 | 0.377 s | 0.155 s | **2.43×** | 99.1% |
| 16 | 2,576 | 0.643 s | 0.170 s | **3.79×** | 99.3% |
| 32 | 5,152 | 1.476 s | 0.171 s | **8.61×** | 99.6% |
| 40 | 6,448 | 2.447 s | 0.192 s | **12.76×** | 99.8% |

不开缓存时，前缀从 1,296 增加到 6,448 token，延迟从 0.377 s 增长到 2.447 s。

开启缓存后，同一范围内延迟只有 0.155～0.192 s。命中的 token 数始终等于共享前缀部分，并且是 16 的整数倍，对应当前配置的 KV block 大小。

进一步固定 16 帧，只改变问题数：

| N | Cache Off | Cache On | 单请求加速 | 整轮加速 |
|---:|---:|---:|---:|---:|
| 5 | 0.633 s | 0.160 s | 3.97× | 1.66× |
| 20 | 0.616 s | 0.135 s | 4.57× | 2.80× |
| 50 | 0.663 s | 0.184 s | 3.61× | 3.17× |

单请求收益没有随问题数量明显变化；问题数量增加主要摊薄了引擎固定开销，因此整轮墙钟收益提高。

这组实验只覆盖串行请求（`max_num_seqs=1`）。并发压力下的缓存淘汰和命中率变化没有测试。伪视频使用重复帧，也不能代表真实视频的视觉编码成本。

---

## 5. SGLang 横向实验

vLLM 使用 Block Hash 管理共享前缀，SGLang 使用 Radix Tree。源码层面最直接的区别如下：

| | vLLM 0.28 | SGLang 0.5.20 |
|---|---|---|
| 索引结构 | Block Hash + 哈希表 | Radix Tree |
| 前缀表示 | 固定 KV Block | 变长 token 序列 |
| 上下文关系 | block hash 包含 parent hash | 树路径表示公共前缀 |
| 匹配粒度 | 16 token | `page_size=1` 时 token 级 |

对应的核心源码分别是：

```python
# vLLM
BlockHash(
    hash_function(
        (parent_block_hash, tuple(curr_block_token_ids), extra_keys)
    )
)
```

和：

```python
# SGLang
prefix_len = child.key.match(key, page_size=self.page_size)

if prefix_len < len(child.key):
    new_node = self._split_node(child.key, child, prefix_len)
```

vLLM 按 block 逐个查询 hash；SGLang 沿 Radix Tree 向下匹配，发生部分匹配时通过 `_split_node()` 将公共前缀拆成独立节点。

为了让两边使用接近的视觉 token 数，SGLang 还需要显式设置视频预处理的 `max_pixels/min_pixels`。未对齐时，同样 8 帧得到：

```text
vLLM:   1296 tokens
SGLang: 3764 tokens
```

对齐后：

```text
vLLM:   prompt 1307 / cached 1296
SGLang: prompt 1306 / cached 1294
```

输入 token 数差约 0.08%。

SGLang 的 Prefix Cache 结果为：

| 帧数 | 前缀 token | Cache Off | Cache On | 加速比 | 命中率 |
|---:|---:|---:|---:|---:|---:|
| 8 | 1,308 | 0.451 s | 0.254 s | 1.78× | 99.1% |
| 16 | 2,596 | 1.715 s | 1.615 s | 1.06× | 99.5% |
| 32 | 5,172 | 2.523 s | 1.848 s | 1.37× | 99.8% |
| 40 | 6,460 | 3.057 s | 1.796 s | 1.70× | 99.8% |

这组绝对延迟不能和 vLLM 横向比较。

原因是本机两边最终使用的 attention backend 不同：

```text
vLLM   → FA2
SGLang → Triton
```

SGLang 默认使用 FlashInfer，但本机系统 CUDA 12.0 与其 JIT 编译要求不兼容，因此实验只能切换至 Triton backend。

同一套 SGLang 配置下，attention backend 已经能带来明显差异：

| backend | 中位延迟 |
|---|---:|
| Triton | 0.216 s |
| torch_native | 0.506 s |
| flex_attention | 0.867 s |

因此这里只保留两个结论：

1. 两边都能获得约 99% 的共享前缀命中率；
2. 当前环境不能通过绝对延迟判断哪套 Prefix Cache 实现更快。

另外，SGLang cache-on 在 16 帧以后出现了与前缀长度相关的额外耗时。目前只有现象，没有 profiler 证据，因此不进一步归因。

---

## 6. 两个影响实验结果的测量问题

第一次 Prefix Caching 实验使用 vLLM 的 `first_token_latency`，结果出现负值：

```text
-2.011 / -1.430 / -1.521
```

源码中该指标由不同来源的时间戳相减得到：

```python
return self.iteration_timestamp - start
```

在当前配置下这套指标不可用，因此后续改用本进程 `perf_counter()` 测墙钟时间。

另一个问题来自 GPU 状态污染。

早期 gmu 扫描曾出现：

```text
0.65 成功
0.70 失败
0.75 失败
0.80 成功
```

这与参数单调性不符。进一步检查发现，同一配置的 `non_kv` 会在 2610 MiB 和 3080 MiB 两个值之间跳变。

异常状态下：

```text
NVML：约 677 MiB 占用，GPU util 100%
CUDA：显示设备基本全空
进程：无
```

单独重复运行 gmu=0.70 后 3/3 均成功，因此问题不在参数，而在前一次失败运行留下的 GPU 状态。

最终的基线检查要求 NVML 和 CUDA 两边同时满足条件：

```python
if nvml <= max_nvml_mib and cuda_free >= min_free_mib:
    return True
```

修复前后 Prefix Caching 的结果差异很大：

| | 修复前 | 修复后 |
|---|---|---|
| Cache On 残留 | 0.338 → 1.562 s | 0.155 → 0.192 s |
| 加速比 | 1.66× → 3.10× | **2.43× → 12.76×** |

因此仓库保留了污染数据，但正式结果只使用通过双重基线检查后的运行。

---

## 7. 限制

当前实验还有几个没有解决的问题：

- Prefix Caching 只测了串行请求，没有测试高并发下的缓存淘汰；
- SGLang 和 vLLM 的 attention backend 不一致，不能比较绝对延迟；
- SGLang cache-on 的额外耗时尚未通过 profiler 定因；
- `max_num_batched_tokens` 只测试了 512 和 1024；
- AWQ 与 NF4 的量化格式和量化范围不完全一致；
- Prefix Cache 使用重复图片构造伪视频，不能代表真实视频的全部预处理开销；
- 当前性能分析还没有 Nsight Systems / Nsight Compute 的 kernel 时间线。

---

## 复现

```bash
# vLLM / HF 环境
bash setup/install_vllm.sh
bash setup/install_bitsandbytes.sh
bash setup/install_accelerate.sh
bash setup/download_qwen2vl_2b.sh
bash setup/gpu_check.sh

# vLLM 三组实验
bash experiments/vlm_inference_benchmark/run_all.sh

# SGLang
python3 -m venv ~/venvs/sglang
~/venvs/sglang/bin/pip install "sglang[all]"
export CUDA_HOME=~/venvs/sglang/lib/python3.12/site-packages/nvidia/cu13

bash experiments/vlm_inference_benchmark/sglang/run_smoke.sh
bash experiments/vlm_inference_benchmark/sglang/run_frames.sh
```

数据与代码：

```text
experiments/vlm_inference_benchmark/       实验代码
results/vlm_inference_benchmark/vllm/      vLLM 原始结果
results/vlm_inference_benchmark/sglang/    SGLang 原始结果
experiments/bnb_kernel_align/              bitsandbytes 4bit 补充实验
```

bitsandbytes 的 4bit kernel 对齐实验与主线关系较弱，因此不在正文展开。该实验主要记录了一点：加载时打印的 alignment warning 不能直接用于判断实际 kernel 路径，最终需要 profiler 确认。
