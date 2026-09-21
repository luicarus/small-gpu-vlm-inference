# bnb_kernel_align — bitsandbytes 4bit GEMM 的快慢路径

**问题**：加载 NF4 量化模型时可能看到这条警告，它到底意味着什么、慢多少？

```text
UserWarning: inner dimension (3420) is not aligned for fast kernel with blocksize=64,
falling back to slower implementation.
```

> 环境：RTX 3050 Ti Laptop（**sm86，20 SM**）· bitsandbytes 0.50.2 · bf16 compute · nf4 / blocksize=64
> 原始数据：`small-gpu-vlm-inference/results/bnb_align/bench_align.jsonl`
> 性质：**纯微基准**，不加载模型，显存占用几十 MB

## 1. 机制：源码里有**三条**回退规则，警告只报告其中一条

`bitsandbytes/backends/cuda/ops.py:934`：

```python
K = A.shape[-1]        # 内维            M = A.numel() // K     # 行数(token 数)
N = shapeB[0]          # 输出维

if M > _gemm_4bit_custom_max_m:      # 规则① CUDA=1536：静默回退（无警告）
    use_custom = False
elif K % blocksize != 0:             # 规则② 内维不对齐：**会打印警告**
    warn("inner dimension ... not aligned ...")
    use_custom = False
else:
    use_custom = _gemm_4bit_use_custom_fn(device_index, dtype, M, N, K)  # 规则③
    # ↑ 按 GPU 架构逐代校准的启发式，**也可能返回 False，同样静默**
```

回退后做什么（`ops.py:904 _dequant_linear_fallback`）：

```python
B_dq = torch.empty(shapeB, dtype=A.dtype, device=A.device)   # 分配 bf16 权重缓冲
_dequantize_4bit_impl(B, absmax, blocksize, quant_type, A.dtype, out=B_dq)  # 反量化
return torch.nn.functional.linear(A, B_dq, bias)             # 普通 GEMM
```

**本机命中的是规则③**（sm86 分支，`ops.py:736`）：

```python
if is_sm86:
    if n_blocks >= num_sms:   return M <= 128    # ← n_blocks=20 = SMs=20 → 命中这条
    if n_blocks >= num_sms//2: return M <= 64
    return M <= 16
```

## 2. 方法：**不能靠「有没有警告」判断走了哪条路**

第一版脚本用「是否打印警告」判定路径 —— **测错了**。因为规则①③都是**静默回退**：
`K=3456`（对齐）不打印警告，但在 `M=512` 时同样走了回退路径。

正确做法：用 `torch.profiler` 抓**真实 kernel 名**判定——

| 路径 | kernel 名特征 |
|---|---|
| bnb 自定义融合 kernel | `gemm_4bit_sm80_m16n8k16` |
| 回退路径 | `kDequantizeBlockwise`（反量化）+ `cutlass::Kernel2` / `ampere_bf16_s16816gemm` |

> **单位校准**（`diag_units.py`）：profiler 的 `device_time_total` 单位是**微秒**。
> 校准依据：同形状下所有 kernel 之和 10.30 ms vs 墙钟 10.76 ms（占 96%）。
> 第一版误乘 1000，数值虚高三个数量级 —— 又一个「测量本身要先被测量」的例子。

## 3. 实测数据（固定 N=1280；重复 3 次，极差 ±0.001~0.009 ms）

| K | K mod 64 | M | **实际路径** | 警告 | kernel 耗时（profiler） | 墙钟 |
|---|---|---|---|---|---|---|
| 3420（不对齐样例） | **28** | 128 | 回退 | ✅ | cutlass **96µs** + 反量化 **63µs** | 0.183 ms |
| 3456 | 0 | 128 | **自定义** | — | gemm_4bit_sm80 **207µs** | 0.189 ms |
| 3420 | **28** | 512 | 回退 | ✅ | cutlass **320µs** + 反量化 63µs | 0.405 ms |
| 3456 | 0 | 512 | 回退 | — | ampere **236µs** + 反量化 63µs | 0.319 ms |
| 3420 | **28** | 1024 | 回退 | ✅ | cutlass 518µs + 反量化 63µs | 0.608 ms |
| 3456 | 0 | 1024 | 回退 | — | ampere 488µs + 反量化 63µs | 0.578 ms |
| 3456 | 0 | 2048 | 回退（规则①） | — | cutlass 992µs + 反量化 64µs | 1.094 ms |

## 4. 四个发现

### 发现 1：那条警告是「红鲱鱼」

`M ≥ 512` 时**不管 K 对不对齐都回退**（规则③要求 M ≤ 128）。
警告报告的是规则②，但"对齐就能走快路径"的暗示**在这个形状下不成立**——两条路径本来就是同一条。

→ **经验**：警告描述的是**判定顺序里最先命中的那条规则**，不等于性能拐点。

### 发现 2：真正的对齐效应在 **cuBLAS 选 kernel**

同为回退路径、形态只差 1%，GEMM kernel 却慢 **35%**：

| K | 被选中的 GEMM kernel | 耗时 |
|---|---|---|
| 3420（不对齐） | `cutlass::Kernel2<cutlass_80_tensor...>` | **320 µs** |
| 3456（对齐） | `ampere_bf16_s16816gemm_bf16_256x128` | **236 µs** |

→ 内维不是 64/128 的整数倍时，底层 GEMM 库会退到**另一套 tiling**。
**这不是 bnb 的问题，是 GEMM kernel 选择的问题**，而它才是唯一可测到的真实对齐代价。

### 发现 3：反量化是恒定的 **63 µs** 入场费

| K | M | 反量化耗时 |
|---|---|---|
| 3420 | 128 | 63 µs |
| 3456 | 512 | 63 µs |
| 3420 | 1024 | 63 µs |
| 3456 | 2048 | 64 µs |

与 K、M 几乎无关。占墙钟比例：**M=128 时 34%，M=1024 时 10%**。

### 发现 4（最有价值）：**这个规模下 4bit 比 bf16 更慢**

| 配置 | bf16 | NF4 | NF4 / bf16 |
|---|---|---|---|
| K=3420, M=128 | 0.106 ms | 0.183 ms | **0.58×** |
| K=3456, M=128 | 0.075 ms | 0.189 ms | 0.39× |
| K=3456, M=512 | 0.240 ms | 0.319 ms | 0.75× |
| K=3456, M=1024 | 0.500 ms | 0.578 ms | 0.87× |
| K=3456, M=2048 | 0.995 ms | 1.094 ms | 0.91× |

**7 个配置全部是 NF4 更慢**，且 **M 越大差距越小**（0.58× → 0.91×）——
大 M 时 GEMM 计算占主导，反量化被摊薄。

> **结论：量化省的是显存，不是时间。**
> compute-bound 区间（大 M / prefill）接近 bf16；latency-bound 区间（小 M / decode）明显更慢。

## 5. 复现

```bash
cd "/mnt/d/Work Places/Python Work Place/Job/InfraStudy/small-gpu-vlm-inference"
bash experiments/bnb_kernel_align/run.sh                              # 约 2 分钟
~/venvs/vllm/bin/python experiments/bnb_kernel_align/diag_units.py    # 重新校准 profiler 单位
```

## 6. 局限与存疑

1. **单一形状**：固定 `N=1280`。其他 N 会改变 `n_blocks`，从而改变规则③的判定阈值——**不可直接外推到别的层**。
2. **单一架构**：只在 sm86（20 SM）上测。源码里 sm80/sm90/sm100 的阈值完全不同。
3. **未做端到端验证**：这是**微基准**，没有测它在整机延迟里占多少。
4. **bf16 对照组的公平性**：`torch.nn.Linear` 走 cuBLAS，而 bnb 自定义 kernel 是另一套实现。
   这个比较回答的是"量化版 vs 未量化版谁快"，**不是**"两种量化方案谁快"。