"""Fused Softmax —— 「融合」为什么能省 IO，以及它距离 FlashAttention 还差什么。

## 这个脚本要回答三个问题

1. **融合（fusion）到底省了什么？**
   朴素实现会把中间结果（`x - max` 的指数矩阵）**写回显存再读回来**；
   融合实现让整行数据在寄存器/SRAM 里走完全程，只读一次、写一次。
   本脚本用「朴素多趟 torch」 vs 「融合（torch.softmax / triton）」把差距量化出来。

2. **Triton 的行内归约怎么写？**
   一个 program 处理一整行：`tl.max(axis=0)` → `exp` → `tl.sum(axis=0)` → 归一化。
   这就是 softmax 的最小完整形态。

3. **它距离 FlashAttention 还差什么？**（脚本末尾有详细对照）
   差的是：**行放不下**。softmax 的一整行能塞进一个 program 的寄存器，所以只需**两趟**；
   而 attention 的 softmax 归一化范围横跨**所有 KV 块**，必须逐块处理 + **在线修正**。
   → 本脚本里那几行「先 max 再 exp 再 sum」就是 FA 在线算法的**雏形**，
      FA 把它升级成了「带着运行最大值循环，并 rescale 已累积的输出」。

## 为什么用 fp32 大矩阵

要让「融合省 IO」这个结论站得住，矩阵必须大到**显著超过 L2 cache**，
否则朴素实现被 cache 兜住、看不出差距。这里用 8192×4096 fp32 = 128 MB。

## 输出什么

三种实现的耗时 / 有效带宽 / 相对最慢者的加速比 + 数值一致性校验。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent))

N_ROWS = 8192
N_COLS = 4096
BLOCK_SIZE = 4096          # 本脚本取「列数向上取整到 2 的幂」，即一行一个 program


@triton.jit
def softmax_kernel(
    out_ptr,                # *fp32  输出矩阵
    in_ptr,                 # *fp32  输入矩阵
    in_row_stride,          # int    输入行距（元素数）
    out_row_stride,         # int    输出行距
    n_cols,                 # int    实际列数
    BLOCK_SIZE: tl.constexpr,
):
    """行内 softmax：一个 program 负责一整行。

    ★★★ 与 FA 的对应关系（读这段时对照 FlashAttention 算法 1）★★★
      - `tl.max(row, axis=0)`  → FA 里的 running max `m_i`
      - `tl.exp(row - m)`      → FA 里的 `exp(S_ij - m_i)`（未归一化的注意力权重）
      - `tl.sum(..., axis=0)`  → FA 里的 running sum `l_i`（归一化因子）
      - 最后 `numerator / denominator` → FA 里的 `O_i / l_i`

    ⚠️ 唯一的关键区别：**这里 max 是"读完整行之后"才知道的**（两趟语义，但都在寄存器里，
    所以不需要真的读两遍）。FA 里一行被切成多个 KV 块，max 必须边算边更新，
    且已累积的输出 O 必须乘 `exp(m_old - m_new)` 被**追溯修正** —— 那才是真正的"在线"。
    """
    row_idx = tl.program_id(axis=0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # ⚠️ 越界 lane 必须填 -inf：否则它们的 0 会参与 max 计算，把最大值拉低。
    #    （FA 里对被 mask 掉的 KV 位置做同样的处理 —— 只不过那里是加 -inf 到分数上）
    row = tl.load(in_ptr + row_idx * in_row_stride + col_offsets,
                  mask=mask, other=-float("inf"))

    row_minus_max = row - tl.max(row, axis=0)      # 数值稳定：减最大值（softmax 平移不变）
    numerator = tl.exp(row_minus_max)              # ★ FA: exp(S - m)
    denominator = tl.sum(numerator, axis=0)        # ★ FA: running sum l
    softmax_out = numerator / denominator          # ★ FA: O / l

    tl.store(out_ptr + row_idx * out_row_stride + col_offsets, softmax_out, mask=mask)


def softmax_naive_torch(x: torch.Tensor) -> torch.Tensor:
    """朴素多趟实现：刻意把中间结果落回显存，用来暴露"融合"省下的 IO。

    每一步都是一次独立的 kernel + 一次显存往返：
        ① x.max()            读 x
        ② (x - m).exp()      读 x、写 num      ← 中间矩阵落盘
        ③ num.sum()          读 num
        ④ num / den          读 num、写 out
    合计约 6 次矩阵级读写；融合实现只需要「读 x + 写 out」2 次 → 理论差距 3×。
    """
    x_max = x.max(dim=1, keepdim=True).values
    num = (x - x_max).exp()
    den = num.sum(dim=1, keepdim=True)
    return num / den


def bench(fn, warmup: int = 10, iters: int = 50) -> float:
    """返回单次调用平均毫秒数。warmup 给够——编译后前几次仍有冷路径。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main() -> None:
    print("=" * 78)
    print("Fused Softmax：融合省的是「中间结果的显存往返」")
    print("=" * 78)
    print(f"  设备    : {torch.cuda.get_device_name(0)}")
    print(f"  矩阵    : {N_ROWS} × {N_COLS} fp32 = {N_ROWS * N_COLS * 4 / 1e6:.0f} MB")
    print(f"  BLOCK   : {BLOCK_SIZE}（一行一个 program）")
    print()

    torch.manual_seed(0)
    x = torch.randn(N_ROWS, N_COLS, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    n_bytes = x.numel() * 4

    # ---- 参考结果（用 torch.softmax 作为数值基准）----
    ref = torch.softmax(x, dim=1)

    # ---- ① 朴素多趟 ----
    t_naive = bench(lambda: softmax_naive_torch(x))
    # ---- ② torch.softmax（C++ 侧已融合）----
    t_torch = bench(lambda: torch.softmax(x, dim=1))
    # ---- ③ Triton 融合 ----
    grid = (N_ROWS,)
    tri = lambda: softmax_kernel[grid](out, x, x.stride(0), out.stride(0), N_COLS,
                                       BLOCK_SIZE=BLOCK_SIZE)
    # 首次调用含 JIT 编译，单独测一次（这是写算子的"warmup 成本"）
    t0 = time.perf_counter()
    tri()
    torch.cuda.synchronize()
    t_compile = (time.perf_counter() - t0) * 1000
    t_triton = bench(tri)

    # 数值校验
    err_torch = (torch.softmax(x, dim=1) - ref).abs().max().item()
    softmax_kernel[grid](out, x, x.stride(0), out.stride(0), N_COLS, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()
    err_triton = (out - ref).abs().max().item()

    # 流量口径（按"一次矩阵级读写"为单位，1 单位 = n_bytes）：
    #   朴素：read x(1) + read x/write num(2) + read num(1) + read num/write out(2) = 6
    #   融合：read x + write out = 2
    rows = [
        ("朴素多趟 torch", t_naive, n_bytes * 6, err_torch),
        ("torch.softmax", t_torch, n_bytes * 2, err_torch),
        ("Triton 融合", t_triton, n_bytes * 2, err_triton),
    ]
    slowest = max(r[1] for r in rows)

    print(f"{'实现':<18} {'ms':>9} {'有效带宽 GB/s':>14} {'相对最慢':>9} {'最大误差':>12}")
    print("-" * 78)
    for name, ms, traffic, err in rows:
        print(f"{name:<18} {ms:>9.4f} {traffic / 1e6 / ms:>14.1f} "
              f"{slowest / ms:>8.2f}× {err:>12.2e}")
    print()
    print(f"  Triton 首次编译耗时: {t_compile:.1f} ms")
    print()
    print("  注：'有效带宽' 按各自理论的显存流量折算 —— 朴素实现流量大 4 倍，")
    print("      所以它即使带宽跑满也更慢；**融合的收益来自「少搬 4 倍数据」，不是算得更快**。")
    print()
    print("=" * 78)
    print("它距离 FlashAttention 还差什么")
    print("=" * 78)
    print("  本脚本的 softmax：**一行 4096 个元素，全部塞进一个 program 的寄存器**")
    print("    → 所以 max / sum 可以『假装两趟、实际一趟』，不需要在线算法。")
    print()
    print("  但 attention 的 softmax 是：")
    print("    out = softmax(Q @ K^T) @ V        ← 归一化范围横跨**整行 K**（可能几万 token）")
    print("    → 行放不下 → 必须切成 KV 块逐块处理 → 于是需要：")
    print("        ① 运行最大值 m 随块更新（本脚本的 tl.max 升级成跨块 max）")
    print("        ② 已累积输出 O 乘 exp(m_old - m_new) 被**追溯修正**（rescale）← 本脚本没有这一步")
    print("        ③ 且 Q@K^T 与 softmax@V **都不落盘**（否则又变回 O(n²) 显存）")
    print()
    print("  → 三件事合起来，就是把『显存从平方降到线性』：")
    print("     不存 N×N 分数矩阵，只存每行的 (m, l) 与输出 O。")
    print()
    print("SOFTMAX_OK")


if __name__ == "__main__":
    main()
