# 4GB 显存下的 VLM 推理实验

模型：`Qwen2-VL-2B-Instruct`（AWQ / bf16）· RTX 3050 Ti Laptop 4GB · WSL2 (Ubuntu 24.04)
vLLM 0.28.0 · SGLang 0.5.20 · transformers 5.16.1 · bitsandbytes 0.50.2 · torch 2.13.0+cu130

**实验分组**：① vLLM 显存边界 → ② vLLM vs Transformers+NFL → ③ vLLM prefix caching
→ ④ 横向对比 SGLang 的 RadixAttention（§5）

**一句话结论**：在 4GB 上，显存预算是**先算后跑**的（§2）；关掉 offload 后 vLLM 比
Transformers + NF4 快 2.5~3.5×（§3）；prefix caching 的收益随前缀长度增长到 **12.76×**（§4）；
SGLang 用基数树做同一件事，但在本机受 attention backend 所限**绝对延迟不可比**（§5）。

---

## 1. 实验目标与环境

### 1.1 我想回答的三个问题

在一张 4GB 显卡上跑 2B 级 VLM，我想搞清楚三件事：

1. vLLM 的显存参数能取到什么范围？越过边界时它是怎么失败的？
2. 在同样装得下的前提下，`vLLM + AWQ` 和 `transformers + bitsandbytes NF4` 差多少？
3. 如果多个请求共享一段很长的前缀（比如同一段视频配不同问题），prefix caching 能省多少？

这三组实验是递进的：第一组给出显存能怎么切，第二、三组都建立在那个预算之上。

### 1.2 为什么用 2B 模型

我最初用 3B 模型做过第一组实验，后来整组重做了，原因是一个实验设计问题：

3B 的 AWQ 权重是 3.17 GiB，而 4GB 卡上 CUDA 实际可用只有 3.21 GiB，
权重还要和激活值、KV cache 抢空间，所以 vLLM 被迫开启 `cpu_offload_gb`。
一旦 offload，解码时每个 token 都要从 CPU 读权重经 PCIe 搬过来
（有效带宽约 7 GB/s，显存约 192 GB/s，差一个数量级），
这时候测出来的性能差距是"谁被迫 offload"造成的，不能归因到引擎本身。

换成 2B（AWQ 权重 2.74 GiB）后 `offload=0` 装得下，这个变量才被消掉。
**本报告全部数据都来自 2B 模型。**

那组 3B 实验的代码和数据保留在 `preliminary_3b_memory_sweep/`，
它记录了"发现混杂变量后重做"的过程。

### 1.3 测量方法

| 项 | 做法 | 原因 |
|---|---|---|
| 显存峰值 | NVML 整卡口径（`nvidia-smi`，100ms 采样） | vLLM 0.28 的推理核心在 EngineCore 子进程，父进程调 `torch.cuda.max_memory_allocated()` 只能读到 0 |
| 延迟 | 报中位数与 P99，不报均值 | 长尾会把均值带偏 |
| 基线 | 每个配置跑前记录，不干净就不接受该次结果 | 见第 6.2 节 |
| 首轮 | 每个新 shape 的第 0 条请求不计入统计 | Triton 运行时 JIT 编译的一次性成本可达数分钟 |
| 解码 | `temperature=0`，`max_tokens` 固定 | 消除采样随机性 |

---

## 2. 实验一：vLLM 的显存边界

### 2.1 实验设计

14 个配置，一次跑一个（CUDA 上下文无法在同一进程里干净复位），
超时用进程组 `killpg` 整组清理（EngineCore 是独立子进程，只杀父进程会留下孤儿继续占显存）。
每个配置从 vLLM 自己的日志里抓显存账本，并给失败分类。

### 2.2 gmu 扫描

| gmu | 预算 MiB | KV 池 | KV tokens | 并发上限 | 状态 |
|---|---|---|---|---|---|
| 0.55 | 2253 | 0 | 0 | — | 失败 |
| 0.60 | 2458 | 0 | 0 | — | 失败 |
| **0.65** | 2662 | 0.07 GiB | 2,544 | 2.48× | 成功 |
| 0.70 | 2867 | 0.27 | 10,032 | 9.80× | 成功 |
| 0.75 | 3072 | 0.47 | 17,520 | 17.11× | 成功 |
| **0.80** | 3277 | 0.67 | 25,008 | 24.42× | 成功 |
| 0.85 | 3482 | — | — | — | 失败 |

可用区间是 **0.65 ~ 0.80**。gmu 每增加 0.05，KV 池约增加 0.20 GiB（约 7,300 tokens）。

三个失败位置对应三种不同的错误，来自源码里不同的检查：

| 失败 | 触发条件 | vLLM 的报错 |
|---|---|---|
| 预算越界 | `gmu × 总显存 > 启动时空闲显存` | `Free memory on device (3.21/4.0 GiB) ... is less than desired GPU memory utilization (0.85, 3.4 GiB)` |
| KV 池为空 | 预算装不下 `non_kv` | `No available memory for the cache blocks` |
| 序列太长 | KV 池存在，但装不下一条 `max_model_len` 的序列 | `To serve at least one request with the model's max seq len (8192), (0.22 GiB KV cache is needed, which is larger than the available (0.2 GiB)` |

第三种报错里 vLLM 会直接给出上限（`the estimated maximum model length is 7440`），照抄即可。

### 2.3 max_model_len 与 max_num_seqs

| 配置 | non_kv MiB | KV 池 | KV tokens |
|---|---|---|---|
| len 1024 | 2610 | 0.67 GiB | 25,008 |
| len 2048 | 2672 | 0.61 | 22,864 |
| len 4096 | 2795 | 0.49 | 18,336 |
| len 8192 | 3041 | 0.25 | 9,184 |
| seqs 2（mnbt 512） | 2580 | 0.70 | 26,048 |
| seqs 4（mnbt 512） | 2611 | 0.68 | 25,312 |
| seqs 8（mnbt 512） | 2611 | 0.68 | 25,328 |

两点值得注意：

- **`max_model_len` 越大，`non_kv` 越大、KV 池越小。** 因为 `max_model_len` 会推高
  profiling 那次 forward 的激活峰值，它和 KV cache 抢同一份预算。
- **把 `max_num_batched_tokens` 显式压到 512 后，并发 2/4/8 的 KV 池几乎不变**（0.68~0.70 GiB）。
  不压的话（默认值由 `max_model_len × max_num_seqs` 推导），batch≥2 会直接因为 KV 池为空而死。

### 2.4 我原来的理解，以及实测后的修正

**实验前**，我把 `gpu_memory_utilization` 理解成"给模型用的显存比例"，
以为调高它主要意味着 OOM 风险增加、调低则更安全。

**实测后**：它首先决定的是 vLLM 的**显存预算**，而预算里要先扣掉一块固定开销——
权重、profiling 峰值激活、CUDA graph，合计 `non_kv`，本机实测 **2610 MiB**。
剩下部分才进入 KV cache。所以：

- 调高 gmu 不是"更危险"，而是**给 KV 池扩容**；
- 调低 gmu 才可能让预算装不下 `non_kv`，**低 gmu 同样会启动失败**（0.60 和 0.55 都失败了）；
- gmu 有一个硬上限 **0.802** = CUDA 可用 3.21 GiB ÷ 总 4.0 GiB，与模型无关，超过必然启动失败。

另外，KV cache 的成本可以算出来：本机 **28.1 KiB/token**，
与架构推算一致（28 层 × 2(K/V) × 2(kv_heads) × 128(head_dim) × 2 B = 28 KiB）。
于是有一个可以手算的判别式：

```
可用 KV 池 = gmu × 总显存 − non_kv
能装下的最长序列 = 可用 KV 池 ÷ 28.1 KiB
```

---

## 3. 实验二：vLLM 与 Transformers 对比

### 3.1 对比条件

OCRBench 100 条小图子集（seed=42），两个引擎用完全相同的输入、同样的贪心解码、
同样的像素预算（`max_pixels=262144`）。batch 取 1/2/4/8，共 800 次推理，零失败。
**全部在 `offload=0` 条件下取得。**

### 3.2 结果

| batch | 引擎 | 运行时长 s | 吞吐 tok/s | 峰值 MiB | P50 s | P99 s | 加载 s |
|---|---|---|---|---|---|---|---|
| 1 | HF + NF4 | 73.46 | 13.7 | **1833** | 0.702 | 1.774 | **16.0** |
| 1 | vLLM + AWQ | **21.20** | **46.1** | 3579 | **0.173** | **0.474** | 73.0 |
| 2 | HF + NF4 | 52.07 | 19.3 | **1899** | 0.459 | 1.118 | **12.9** |
| 2 | vLLM + AWQ | **17.60** | **55.6** | 3579 | **0.146** | **0.350** | 63.8 |
| 4 | HF + NF4 | 34.38 | 29.4 | **2065** | 0.269 | 0.613 | **15.4** |
| 4 | vLLM + AWQ | **13.29** | **73.6** | 3579 | **0.102** | **0.279** | 63.7 |
| 8 | HF + NF4 | 28.61 | 35.4 | **2249** | 0.353 | 0.518 | **16.4** |
| 8 | vLLM + AWQ | **11.25** | **86.9** | 3605 | **0.088** | **0.219** | 61.2 |

随 batch 从 1 增大到 8，HF 的总运行时间从 73.46 s 降到 28.61 s，vLLM 从 21.20 s 降到 11.25 s。
**vLLM 始终更快，但优势从 3.5× 收窄到 2.5×。**

准确率（按 OCRBench 真值匹配）：batch=1 时两边都是 85%，batch=4 时 HF 84%、vLLM 85%。
差异 1 个百分点，n=100 下不显著，因此准确率只能作为回归检查，不能作为优劣证据。

### 3.3 显存曲线为什么不同

| batch | HF + NF4 | vLLM + AWQ |
|---|---|---|
| 1 | 1833 MiB | 3579 MiB |
| 8 | 2249 MiB（+416） | 3605 MiB（+26） |

HF 的峰值随 batch 增长，因为它要给每一路请求留出 padding 与激活；
vLLM 几乎不变，因为 PagedAttention 的 KV 池在启动时一次性预分配好了（本机 0.67 GiB），
之后批量增大只是消费池子里已有的块。

代价也在这里：vLLM 在 batch=1 时就占掉 3579 MiB，比 HF 多 **1.71 GiB**。
这 1.71 GiB 换来的是 2.5~3.5× 的吞吐，以及可预测的并发能力。

另外，**batch=8 时 HF 的单条延迟反而回升了**（P50 0.269 → 0.353 s），vLLM 继续下降（0.102 → 0.088 s）。
因为静态批要把一批里所有序列 padding 到等长，且整批要等最慢的一条；
vLLM 是迭代级调度，完成即退出。

### 3.4 这个实验能说明什么、不能说明什么

**能说明**：同样装得下的前提下，vLLM 的吞吐优势是真实的，且在 batch=1 时最大（3.5×）。
它的显存开销换取的是并发能力，这是一个明确的工程取舍。

**不能说明**：

- **不能说明"vLLM 一定更快"。** 前提是权重装得下。被迫 offload 时结论会反过来——
  我自己在 3B 上的对照实验里，vLLM 落后 1.8×。
- **不能说明哪种量化更好。** 两边量化范围并不对称：vLLM 侧 ViT 保持 bf16
  （AWQ 的默认行为，`modules_to_not_convert=["visual"]`），HF 侧 ViT 也被量化为 NF4。
  这会让 HF 的显存优势被放大（约 1.3 GiB），速度对比受影响较小（ViT 只在 prefill 用一次）。
  想对齐这条路走不通：`transformers 5.16.1 + bnb 0.50.2` 在混精度下抛 `AssertionError`，
  实测与显存无关（2B 上峰值仅 2.6 GiB 仍失败）。
- **两边的量化格式不同**：AWQ 是预量化权重，NF4 是运行时量化。
  vLLM 0.28 的量化后端清单里没有 bitsandbytes，所以"vLLM × NF4"这个组合做不出来。
- **prompt 模板不同**：HF 走 `apply_chat_template`，vLLM 手写模板。
  跨引擎的答案一致率因此偏保守：严格口径 46%、语义口径 56%，
  但人工检视发现差异绝大多数是表述风格（`says "X"` vs `reads "X"`），不是实质分歧。

---

## 4. 实验三：Prefix Caching

### 4.1 实验设计

prefix caching 只在多个请求共享同一段前缀时才有收益。用 100 张各不相同的图（如 OCRBench）
时共享部分接近零，测不出东西。所以我构造了一个专用负载：
**同一张图重复成 K 帧当伪视频**作为共享前缀，配 N 个不同问题。

对照组用 `--no-enable-prefix-caching` 显式关闭。
**vLLM 0.28 里这个特性默认是开启的**，不显式关掉就是"开 vs 开"，测不出任何差异。

### 4.2 前缀长度实验

固定 N=10，只改帧数：

| 帧数 | 前缀 token | 缓存关（中位） | 缓存开（中位） | 加速比 | 命中率 |
|---|---|---|---|---|---|
| 8 | 1,296 | 0.377 s | 0.155 s | **2.43×** | 99.1% |
| 16 | 2,576 | 0.643 s | 0.170 s | **3.79×** | 99.3% |
| 32 | 5,152 | 1.476 s | 0.171 s | **8.61×** | 99.6% |
| 40 | 6,448 | 2.447 s | 0.192 s | **12.76×** | 99.8% |

拆开看两组数：

| 帧数 | 省下 (关−开) | 每 token 节省 | 缓存开后残留 |
|---|---|---|---|
| 8 | 0.222 s | 0.172 ms | 0.155 s |
| 16 | 0.473 s | 0.184 ms | 0.170 s |
| 32 | 1.304 s | 0.253 ms | 0.171 s |
| 40 | 2.255 s | 0.350 ms | 0.192 s |

### 4.3 结果解释

关闭缓存时，延迟随前缀增长上升得越来越快（0.377 → 2.447 s），
每 token 的成本从 0.172 ms 涨到 0.350 ms。这与注意力的计算量随序列长度平方增长相符：
序列越长，摊到每个 token 上的 prefill 成本越高。

开启缓存后，请求延迟保持在 0.155 ~ 0.192 s，帧数涨 5 倍而延迟只涨 1.24 倍。
这部分是缓存覆盖不到的开销（视觉编码 + decode），实测它基本上不随前缀长度增长。

两者相除就是加速比：前缀从 1,296 涨到 6,448 token（5.0 倍）时，加速比从 2.43× 涨到 12.76×（5.3 倍）。

命中率数据从另一侧印证了机制：**缓存命中的 token 数恒等于前缀长度**（且总是 16 的整数倍，
对应 vLLM 的块大小），不随问题长度变化——说明复用是前缀级的，每个请求只重算自己的尾巴。

### 4.4 补充：问题数 N 的影响

固定 16 帧（前缀 2,576 token），只改问题数：

| N | 缓存关 中位 | 缓存开 中位 | 单请求加速比 | 整轮加速比 |
|---|---|---|---|---|
| 5 | 0.633 s | 0.160 s | 3.97× | 1.66× |
| 20 | 0.616 s | 0.135 s | 4.57× | 2.80× |
| 50 | 0.663 s | 0.184 s | 3.61× | 3.17× |

单请求的加速比基本不随 N 变化（每个请求各自付一次 prefill），
但整轮墙钟的加速比随 N 增长，因为引擎的固定开销被更多请求摊薄。

所以收益来自"前缀大"，不是"问题多"。

### 4.5 当前结论的适用范围

- 这是**单请求串行**（`max_num_seqs=1`）的结果。并发时共享前缀块可能被淘汰——
  第一波请求完成后，它们持有的前缀块引用计数归零，池子紧张时会被 LRU 清掉。
  这需要一个专门的并发实验来测，目前没做。
- 伪视频的帧是同一张图重复。这**不影响缓存结论**（前缀哈希只取决于 token 内容是否一致），
  但真实视频的视觉编码成本可能不同。
- 输出长度固定为 16 token，decode 部分在各组间是常数，便于比较；
  真实问答输出更长，decode 占比更高。
- 本机 KV 池只够 `max_model_len ≤ 7440`，所以 48 帧（7,776 token 前缀）做不了。
  想测更长的前缀需要更大的显存。

---

## 5. 横向对比：SGLang 的 RadixAttention

**问题**：实验三证明了 prefix caching 的收益。但那是 vLLM 的实现。
SGLang 用**完全不同的数据结构**做同一件事，两者的差异在哪？
在同一份负载上跑一遍，才能把差异归因到机制而不是负载。

### 5.1 两种数据结构的差异（读源码得出）

| | vLLM 0.28 | SGLang 0.5.20 |
|---|---|---|
| 索引结构 | **块哈希链 + 哈希表** | **基数树（radix tree）** |
| 复用单位 | **固定 16-token 块** | **变长 token 序列（树的一条边）** |
| 前缀身份 | 块哈希**链上父块哈希** | **树路径本身** |
| 对齐损失 | 不满 16 token 的尾巴不能复用 | 无（`page_size` 可为 1） |

**根因**：vLLM 用哈希**找回**一段前缀，所以必须把它切成可寻址的固定块；
SGLang 用树**走**到一段前缀，路径天然带上下文，不需要固定粒度。

两边的机制细节：

```python
# vLLM：kv_cache_utils.py:598 —— 块的"身份"包含它的全部历史
return BlockHash(hash_function((parent_block_hash, tuple(curr_block_token_ids), extra_keys)))

# SGLang：radix_cache.py:724 —— 沿树下降，走到边的中间就分裂
prefix_len = child.key.match(key, page_size=self.page_size)
if prefix_len < len(child.key):
    new_node = self._split_node(child.key, child, prefix_len)
```

两个我在读源码时没预料到的点：

- **SGLang 的 `page_size` 是可调的取舍**。`RadixKey.match()` 的 docstring 写明
  *"Result is **rounded down to `page_size`**"* —— 即调大它就是**主动引入对齐损失来换取更快的匹配**。
  而 vLLM 的块大小是启动时固定、运行中不可调的。
- **淘汰策略在 SGLang 里是可插拔的**（`evict_policy.py` 有 `LRUStrategy` / `LFUStrategy` / `FIFOStrategy`），
  且淘汰是**对可淘汰叶子建优先队列、自底向上**进行——一次释放一整段 token 序列，不是固定块。

> **本实验的负载下，这个机制差异会被摊薄**：几千 token 的长前缀把 16-token 的对齐损失
> 压到 ≤15/6448 ≈ **0.2%**。RadixAttention 的优势场景是**短前缀 + 高分叉**
> （典型：多轮对话里大量请求共享一小段 system prompt 后各自展开）。

### 5.2 本机把 SGLang 跑起来（踩了五个坑）

在 4GB 卡的 sm86 上让 SGLang 0.5.20 跑起来，比预期费劲：

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | `Loaded weights leave no GPU memory for the KV cache` | `mem_fraction_static=0.75` 时静态池装不下权重 | SGLang 自己报出最小可用 0.7508，改 0.80 |
| 2 | `nvcc fatal: Unknown option '--compress-mode=size'` | flashinfer JIT 需要 CUDA 12.1+，**系统 nvcc 是 12.0** | `CUDA_HOME` 指向 venv 自带的 CUDA 13.4 |
| 3 | `CUDA compiler and CUDA toolkit headers are incompatible` | 换 nvcc 后与 flashinfer 自带的 CCCL 头文件版本冲突 | **改用 `--attention-backend triton`**，绕开 flashinfer |
| 4 | `Unsupported video input type: PIL.Image` | 嵌套层级：SGLang 的多模态输入**按请求分组** | 传 `[[f1..fn]]` 而非 `[f1..fn]` |
| 5 | `input (3764 tokens) is longer than context length` | 视频每帧像素预算不一致（见 5.3） | 对齐预算 |

**第 3 条最值得记**：本机系统 CUDA（12.0，2023 年）与 SGLang 0.5.20 所需 flashinfer
存在**根本性版本差**。最终靠换 attention backend 绕开，而不是装新 CUDA ——
后者会牵动已经跑通的三组实验环境。

另外必须**独立 venv**：SGLang 要的 `transformers` 是 5.12.1，而我们 vLLM 环境是 5.16.1，
混装会**静默改变依赖树**、让已有结论失效。

### 5.3 口径对齐：一个差点让对比失效的问题

移植后第一次跑，同一张图、同样 8 帧，两边的 prompt token 数**差 2.9 倍**：

| | prompt token |
|---|---|
| vLLM（`max_pixels=262144`） | 1,296 |
| SGLang（默认） | **3,764** |

**根因**：SGLang 对图片与视频用**两套默认预算**（`qwen_vl.py:59-75`）：

```python
MAX_PIXELS         = 16384*28*28 = 12,845,056   # 图片
VIDEO_MAX_PIXELS   = 768*28*28   =    602,112   # 视频每帧（硬编码常量）
VIDEO_TOTAL_PIXELS = env VIDEO_MAX_PIXELS or ... # 读环境变量
# 视频每帧预算 = max(min(VIDEO_MAX_PIXELS, TOTAL/nframes*2), MIN*1.05)
```

**踩坑**：我先试了环境变量 `VIDEO_MAX_PIXELS`，**完全无效**。读三层源码才找到真正生效的路径：

```
qwen_vl.py:69       VIDEO_TOTAL_PIXELS = os.environ.get("VIDEO_MAX_PIXELS", ...)
                    ← 模块导入时求值，但**不是实际生效路径**
qwen_vl.py:1027     await preprocess_video(video, video_config=self.video_config)
                    ← 真正生效的是这个
base_processor.py:269   self.video_config = mm_process_config.get("video", {})
                    ← 来自 server 参数，**覆盖模块常量**
```

正解是 `Engine(mm_process_config={"video": {"max_pixels": ..., "min_pixels": ...}})`，
等价于 vLLM 的 `mm_processor_kwargs`。

**对齐后**：

| | prompt token | cached token |
|---|---|---|
| vLLM | 1,307 | 1,296 |
| SGLang | **1,306** | 1,294 |

**差 0.08%** ✅

### 5.4 结果

| 帧数 | 前缀 token | 缓存关（中位） | 缓存开（中位） | 加速比 | 命中率 |
|---|---|---|---|---|---|
| 8 | 1,308 | 0.451 s | 0.254 s | **1.78×** | 99.1% |
| 16 | 2,596 | 1.715 s | 1.615 s | **1.06×** | 99.5% |
| 32 | 5,172 | 2.523 s | 1.848 s | **1.37×** | 99.8% |
| 40 | 6,460 | 3.057 s | 1.796 s | **1.70×** | 99.8% |

先确认命中数指标可比：SGLang 的 `meta_info["cached_tokens"]` 定义在
`schedule_batch.py:1240`（*"The number of cached tokens that were already cached in the KV cache"*），
赋值是 `pre_len - already_computed`，其中 `pre_len = len(prefix_indices)`
（调度器准入时认领的 KV slot）。**语义与 vLLM 的 `num_cached_tokens` 一致** ✅

与 vLLM 同负载并排：

| 帧数 | vLLM 加速比 | SGLang 加速比 |
|---|---|---|
| 8 | 2.43× | 1.78× |
| 16 | 3.79× | **1.06×** ← 凹陷 |
| 32 | 8.61× | 1.37× |
| 40 | 12.76× | 1.70× |

**趋势不同**：vLLM 单调快速上升，SGLang 非单调。

### 5.5 ⚠️ 为什么不能比较绝对延迟

这是本实验**最重要的限制**。两个引擎在这台机器上的 attention backend **不对等**：

| | 引擎默认 | 本机实际 |
|---|---|---|
| vLLM | `FLASH_ATTN` | ✅ **FA2**（自带预编译的 `_vllm_fa2_C.abi3.so`，不依赖系统 nvcc） |
| SGLang | `flashinfer` | ❌ 装不起来（见 5.2）→ 只能用 `triton` |

而 **backend 选择对性能的影响极大**（同一引擎、同一负载、其余参数完全一致）：

| backend | 中位延迟 | vs triton |
|---|---|---|
| `triton` | **0.216 s** | 1.00× |
| `torch_native` | 0.506 s | **2.34×** |
| `flex_attention` | 0.867 s | **4.01×** |

**同一引擎换 backend 差 4 倍。** 所以：

> **绝对延迟不可横向比较**，只能比较**趋势与命中行为**。
> 本节的 1.78/1.06/1.37/1.70× 是「在 triton 后端、4GB 显存约束下」的观测值，
> **不能读成"SGLang 的前缀复用更差"**。

顺带一个发现：SGLang **FA2 的 kernel 其实在 sm86 上可用**
（`sgl_kernel` 自带，实测 `flash_attn_varlen_func` 调用成功），
缺的只是注册表入口 —— 0.5.20 没有 `@register_attention_backend("fa2")`。
强行注册需要改 site-packages，**为保证他人能复现，没有采用**。

### 5.6 一个仍未定因的现象

SGLang 在 cacheon 下，耗时与前缀长度**超线性**，且与缓存命中无关：

| 配置 | 前缀 token | **未缓存** | 中位 wall |
|---|---|---|---|
| 8 帧 N=3 | 1,308 | **12** | **0.263 s** |
| 16 帧 N=3 | 2,596 | **12** | **1.543 s** |

**未缓存 token 相同（12 vs 12），耗时差 5.9×**；且与请求数无关（N=3 vs N=10 仅差 13%）。
作为对照，vLLM 在同样两档下是 0.155 → 0.170 s（**1.10×**）。

我原本推断是"KV 池装不下导致请求间互相驱逐"，**实验推翻了它**：
把 N 从 10 降到 5（需求 12,980 < 池 14,274，装得下），加速比反而从 1.06× → **0.95×**。

**当前状态：现象确凿，成因是推断，没有 profile 证据。**
日志里有这条警告指向 token pool 写入：

```
Triton kernel 'write_req_to_token_pool_triton' device-loaded after serving started
(free device mem: 0.00 GiB). Pre-load it during engine init to avoid CUDA OOM.
```

每个请求都要把 `prefix_indices`（长度 = 前缀 token 数）写进 token pool，
该操作规模 ∝ 前缀长度且与是否命中无关。但**要坐实必须抓 kernel 时间线**，
本节只记录现象，不写成结论。

### 5.7 这一节能说明什么、不能说明什么

**能说明**：

1. **两种前缀复用机制的数据结构差异是真实的**（读源码确认，非传闻）。
2. **跨引擎口径可以对齐**：对齐像素预算后视觉 token 数差 0.08%。
3. **命中数指标语义一致**，可以对比。
4. **引擎的可部署性存在真实差异**：vLLM 自带预编译 FA2 kernel，
   SGLang 在本机受系统 CUDA 版本所限只能用退化后端。这是一个**工程事实**，
   而不是"谁更快"的结论。
5. **backend 敏感度被量化了**（同一引擎 4 倍差距）——这条对任何
   跨引擎 benchmark 都是提醒：**不固定 backend 的对比没有意义**。

**不能说明**：

1. ❌ 不能比较绝对延迟（backend 与 KV 池都不对等）。
2. ❌ 不能说 SGLang 的前缀复用更差（5.6 的成因未确认）。
3. ❌ 不能外推到其他硬件（backend 可用性正是本机 CUDA 版本决定的）。

### 5.8 复现

```bash
# SGLang 需要独立 venv（它要的 transformers 版本与 vLLM 不同）
python3 -m venv ~/venvs/sglang
~/venvs/sglang/bin/pip install "sglang[all]" -i https://mirror.sjtu.edu.cn/pypi/web/simple

# 必须：让 JIT 用 venv 的 CUDA 13.4，而不是系统的 12.0
export CUDA_HOME=~/venvs/sglang/lib/python3.12/site-packages/nvidia/cu13

cd experiments/vlm_inference_benchmark/sglang
bash run_frames.sh
```

⚠️ **注意 `mem_fraction_static` 的语义与 vLLM 的 `gpu_memory_utilization` 相反**：

```python
# kv_cache_configurator.py:2165
slack_gb = pre_model_load_memory * (1 - mem_fraction_static)
rest_memory = available_gpu_memory - slack_gb - mm_reservation_gb   # ← 这才是 KV 预算
```

它是「**有意留空（非 KV）的比例**」—— 调大它，KV 池才变大。
实测：mfs=0.80 → KV 池 2,302 token；0.90 → 14,274；0.92 → 16,669。
且**长 context 需要更高 mfs**（长 context 抬高 profiling 的激活峰值）：
ctx=6144 时 0.90 直接启动失败，需 0.92；ctx=8192 需 0.93。

---

## 6. 实验过程中修正的两个错误

这两处都不是实验设计问题，而是**测量工具本身出错**，而且错误的数据看起来完全正常。

### 6.1 `first_token_latency` 指标不可用

第一版实验三用 vLLM 的 `RequestOutput.metrics.first_token_latency` 作为主指标，
结果出现了**负数**（-2.011 / -1.430 / -1.521），物理上不可能。

查源码（`v1/metrics/stats.py:373`）：

```python
def _time_since(self, start: float) -> float:
    return self.iteration_timestamp - start   # 引擎核心迭代时钟 − arrival_time（事件时间戳）
```

两个时间戳来自不同来源，相减可以为负。同一条链路上的 `prefill_time` / `e2e_latency`
恒为 0（事件未填充），说明这套指标在本配置下整体不可用。

改用本进程 `perf_counter` 墙钟后，同一批配置的结果从
`1.23 / 1.80 / 1.12 / 4.11` 变成单调的 `1.66 / 1.86 / 2.12 / 3.10`。
（后面这组数字在第 6.2 节里又被进一步修正。）

### 6.2 GPU 状态污染导致了一次错误结论

**这一节是本报告最值得记录的部分。**

**现象。** 实验一第一次扫描出现了物理上不可能的结果：
gmu 0.65 成功、**0.70 失败**、0.75 失败、0.80 成功——预算是单调递增的，中间那档不该失败。

**定位。** 我怀疑过是参数本身的问题，但先检查了一个更基础的东西：
失败时 `non_kv` 被测成 **3080 MiB**，而成功时是 **2610 MiB**。
`non_kv` 是权重 + 激活 + CUDA graph，在相同配置下应该是个常数。
它出现两簇值，说明测量的环境不是同一个。于是我把 gmu 0.70 单独连跑 3 次：
**3/3 全部成功，KV 恒为 0.27 GiB**。这确认了失败与参数无关，是环境问题。

**根因。** GPU 进入了一种不一致的状态：

```
NVML:  677 MiB 已用 / util 100% / 84°C    ← 显示有东西在占用
CUDA:  可用 3287 MiB（"全空"）             ← 但 CUDA 认为设备是空的
进程:  空（WSL 与 Windows 侧都查不到进程）
```

没有任何进程，两套口径却给出相反的答案。紧接在失败配置之后出现时，它会
让下一个配置的 vLLM 量到偏大的 `non_kv`（2610 → 3080 MiB）从而被判失败；
同时让同一配置的单请求耗时出现双峰（例如 0.15 s 与 2.3 s 交替）。

**修复过程。** 这里我走了两次弯路：

1. 我先把等待条件从 `nvidia-smi` 改成 CUDA 的 `mem_get_info()`——
   因为之前遇到过"驱动回收滞后，nvidia-smi 已显示干净但 CUDA 还没拿回空间"。
   但这次 CUDA **报全空**，门槛永远通过，等于没修。
2. 反过来只查 NVML 也不行——回到第 1 条那个场景就失效了。

最终改成**两个口径同时满足才算干净**：

```python
def wait_for_memory_release(min_free_mib=3050, max_nvml_mib=300, timeout_s=90):
    while ...:
        nvml, cuda = gpu_used_mib(), cuda_free_mib()
        if 0 <= nvml <= max_nvml_mib and cuda >= min_free_mib:
            return True, "..."
        time.sleep(3)
    return False, "..."     # 超时就明确报告"该配置可能被污染"
```

**被推翻的结论。** 修复前的那一批实验三数据给出了一个看起来很自洽的结论：

> 缓存覆盖不到的部分（视觉塔编码 + decode）稳定在 39 ms/帧，随帧数线性增长，
> 因此单靠 prefix caching 无法解决长视频推理。

它甚至能解释为什么加速比增长慢于前缀长度。我把它写进了报告初稿。

干净重测后：

| | 修复前 | 修复后 |
|---|---|---|
| 缓存开后残留 | 0.338 → 1.562 s（随帧数线性涨） | 0.155 → 0.192 s（近似恒定） |
| 残留/帧 | 39 ~ 42 ms | 19.4 → 4.8 ms |
| 加速比 | 1.66× → 3.10× | **2.43× → 12.76×** |

那个"39 ms/帧"并不存在。它是 GPU 异常状态给每个请求加的固定开销，
被我在分析里误读成了"每帧成本"。真实情况是视觉编码并不是瓶颈。

**这次经历中我印象最深的一点**：被污染的数据不会显得"很乱"，它会显得**很有规律**。
因为它叠加的是一个系统性偏置而不是随机噪声。分位数、方差、重复测量都发现不了它。
我这次能发现，靠的是两条：一是注意到 `non_kv` 这个物理常数在跳变；
二是把可疑配置单独重跑复现。"每个配置之间验证环境"比"跑完再解释异常"便宜得多。

---

## 7. 局限与下一步

1. **并发下的 prefix caching 没测。** 4.5 节列出的淘汰问题需要一个专门的并发实验
   （同一条共享前缀 + 多路并发），目前只有串行数据。
2. **跨引擎对比的 backend 不对等**（5.5 节）。这是本次最想解决但没解决的限制：
   SGLang 在本机只能用退化后端，所以 §5 只能比趋势，不能比绝对延迟。
3. **SGLang cacheon 的超线性耗时成因未确认**（5.6 节）。现象确凿、假设被实验推翻，
   但缺 profile 证据。这是最值得继续挖的一条。
4. **`max_num_batched_tokens` 只测了两档**（512 / 1024）。
   它在 batch≥2 时是生死开关，但和并发度的交互没有系统扫描。
5. **量化范围不对称**与**量化格式不同**（3.4 节），受框架与引擎能力限制，本机消不掉。
6. **伪视频不是真视频**，可能低估真实视频的视觉编码成本。
7. **单机单次会话**，速度结论没有 Nsight profile 支撑。

下一步有两个方向：

- **补 profile**：用 `torch.profiler` / nsys 抓 SGLang cacheon 路径的 kernel 时间线，
  坐实 5.6 的成因。这是当前唯一一条"有现象、无解释"的结论。
- **并发容量对比**：在同样的显存下两个引擎各能塞下多少并发请求，
  以及并发压力下共享前缀的命中率会不会下降。

---

## 附录 A：复现命令

```bash
# 环境
bash setup/install_vllm.sh
bash setup/install_bitsandbytes.sh
bash setup/install_accelerate.sh
bash setup/download_qwen2vl_2b.sh
bash setup/gpu_check.sh

# 三组实验（串行，约 45 分钟）
bash experiments/vlm_inference_benchmark/run_all.sh

# 或单独跑
bash experiments/vlm_inference_benchmark/vllm/exp1_memory_boundary/run.sh
bash experiments/vlm_inference_benchmark/vllm/exp2_engine_comparison/run.sh
bash experiments/vlm_inference_benchmark/vllm/exp3_prefix_caching/run_frames.sh

# 打印汇总表（只打印，不落盘）
python experiments/vlm_inference_benchmark/vllm/exp1_memory_boundary/summarize.py
python experiments/vlm_inference_benchmark/vllm/exp2_engine_comparison/compare.py

# --- 横向对比（§5）：SGLang 需要独立 venv + 自己的环境变量，见 5.8 节 ---
python3 -m venv ~/venvs/sglang
~/venvs/sglang/bin/pip install "sglang[all]" -i https://mirror.sjtu.edu.cn/pypi/web/simple
export CUDA_HOME=~/venvs/sglang/lib/python3.12/site-packages/nvidia/cu13
bash experiments/vlm_inference_benchmark/sglang/run_smoke.sh   # 先冒烟
bash experiments/vlm_inference_benchmark/sglang/run_frames.sh
```

Python 依赖装在 venv（`~/venvs/vllm`）里，不在系统解释器中——
vLLM 对 torch/CUDA 构建版本有要求，而 Ubuntu 24.04 禁止系统级 `pip install`（PEP 668）。
脚本内部通过 `shared/paths.sh` 自动解析解释器路径。

## 附录 B：数据目录

| 内容 | 位置 |
|---|---|
| 实验代码 | `experiments/vlm_inference_benchmark/` |
| vLLM 原始数据（JSONL + 日志） | `results/vlm_inference_benchmark/vllm/` |
| **SGLang 原始数据** | `results/vlm_inference_benchmark/sglang/` |
| SGLang 实验说明（环境前提/怎么读） | `experiments/vlm_inference_benchmark/sglang/README.md` |
| 被取代的 3B 前置实验 | `experiments/vlm_inference_benchmark/vllm/preliminary_3b_memory_sweep/` |
| 测试图说明（换图影响） | `assets/README.md` |

`results/` 与 `experiments/` 同名同构，映射是计算出来的而不是写死的。
`experiments/vlm_inference_benchmark/` 内**按引擎分组**（`vllm/` 与 `sglang/` 并列）。

仓库里保留了几批跑废的数据，它们是第 6 节各条结论的证据：

| 目录 | 内容 |
|---|---|
| `exp1_memory_boundary/_contaminated_first_run/` | 异常状态下的第一次扫描，含"gmu 0.70 单独复跑 3 次全部成功"的证据 |
| `_ghost_state_run/` | 换测试图后、修复门槛前的那一轮 |
| `exp3_prefix_caching/_bad_metric/` | 用 `first_token_latency` 作主指标的那一版（含负延迟） |
| `_old_image/` | 换测试图之前的全部实验三数据 |

（后两个位于 `results/vlm_inference_benchmark/vllm/` 下，内部按实验名再分一层。）

## 附录 C：bitsandbytes 4bit kernel 的补充实验

这个实验和 VLM benchmark 主线关系不大，是在排查 HF NF4 加载时的告警时顺手做的，
单独放在附录。

**起因**：加载 NF4 模型时 bitsandbytes 报
`inner dimension (3420) is not aligned for fast kernel with blocksize=64, falling back to slower implementation`。

**发现 1：这条告警不等于性能拐点。** 读源码（`bitsandbytes/backends/cuda/ops.py:934`）发现
bnb 有三条回退规则，告警只报告其中一条（内维不对齐），另外两条（`M` 过大、
按架构校准的启发式）都是静默回退。实测 `M ≥ 512` 时不管内维是否对齐都会回退，
所以"对齐内维就能走快路径"在这个形状下并不成立。

我第一版脚本用"有没有打印告警"判断走了哪条路，这是错的，必须用 `torch.profiler`
抓真实 kernel 名。

**发现 2：真正的对齐效应在 cuBLAS 选 kernel。** 同为回退路径，内维不对齐时被选中的
GEMM kernel 慢 35%（320 µs vs 236 µs）。

**发现 3：反量化是恒定的约 63 µs 入场费**，与 K、M 几乎无关。

**发现 4：这个规模下 4bit 比 bf16 更慢。** 7 个配置全部如此，比值 0.58× ~ 0.91×，
且 M 越大差距越小。**量化省的是显存，不是时间。**

完整数据与脚本：`experiments/bnb_kernel_align/`（RTX 3050 Ti / sm86 / N=1280 下测得，
结论不可外推到其它 N 或架构）。
