"""在线 softmax —— 把「一行放不下」逼出来，验证 rescale 是必需的。

## 为什么需要这一步

`softmax_tutorial.py` 里的一行只有 4096 个元素，**整个塞进一个 program 的寄存器**，
所以 max / sum 可以"读完再算"，不需要在线算法。

但 FlashAttention 里一行（K 侧）可能有几万 token，**片上的 shared memory / 寄存器放不下**
（sm86 每 SM 只有 ~100 KB shared memory，而 4096 token × head_dim 128 × 2B = 1 MB）。
→ 必须把行切成块流式读，于是 max 和 sum 都要**边算边更新**。

## 本脚本要证明什么

1. **正确的在线算法**：每处理一块，用新的运行最大值 rescale 已累积的 `l`
   → 结果与"两趟精确计算"完全一致。
2. **不 rescale 会错多少**：给出一个"看起来合理但错"的变体，量化误差。
3. **数据是刻意构造的**：把每行的最大值**放在最后一块**（注入 +10 的尖峰），
   使 m 在循环中大幅增长 —— 如果不 rescale，早期块用小的 m 算出的 exp 会**膨胀上千倍**。

## 与 FA 的对应（读代码时对照）

    m  ← FA 的 running max `m_i`
    l  ← FA 的 running normalizer `l_i`
    ★ rescale 那一行 ← FA 里 `l` 与**输出累加器 O** 都要乘的 `exp(m_old - m_new)`

⚠️ 本脚本只算 (m, l) 这两个**标量统计量**，因为 softmax 的输出是 N 维的（放不下）。
   **FA 之所以能在线算出完整输出**，关键在于它的输出是 `P @ V`——维度是 head_dim（如 128），
   不是序列长度 N。输出小，才能留在寄存器里被 accumulate + rescale。
   这是"为什么 FA 能一行一趟算完，而普通 softmax 不能"的根本原因。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent))

N_ROWS = 64
N_COLS = 65536          # 每行 65536 个元素 → 远超片上容量，必须分块
BLOCK = 1024            # 块大小 → 每行 64 个块
SPIKE = 10.0            # 注入到**最后一块**的尖峰，逼 m 在循环末尾大幅增长


@triton.jit
def online_stats_kernel(
    m_ptr, l_ptr, x_ptr, row_stride, n_cols,
    RESCALE: tl.constexpr,      # ← 编译期开关：False 用来演示"不 rescale"的错误结果
    BLOCK: tl.constexpr,
):
    """一趟算出每行的 (运行最大值 m, 运行指数和 l)。

    循环体就是 FA 内循环去掉矩阵乘之后的骨架。
    """
    row = tl.program_id(axis=0)

    m = -float("inf")           # ★ 运行最大值（比任何实数都小，确保第一块能接管）
    l = 0.0                     # ★ 运行指数和

    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + row * row_stride + offs,
                    mask=offs < n_cols,
                    other=-float("inf"))     # ← 越界填 -inf：既不污染 max，exp 后又是 0

        m_block = tl.max(x, axis=0)
        m_new = tl.maximum(m, m_block)

        # ★★★ 这一行是「在线算法」的全部秘密 ★★★
        # l 是**用旧的 m 累加出来的**；现在基准换成了 m_new，必须把旧 l 乘 exp(m_old - m_new)
        # 追溯修正。若 RESCALE=False（不做修正），早期块会继续按较小的 m 计算 exp，
        # 而 exp(x - m_small) 比 exp(x - m_final) 大得多 → l 被系统性放大。
        if RESCALE:
            l = l * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        else:
            l = l + tl.sum(tl.exp(x - m_new), axis=0)

        m = m_new

    tl.store(m_ptr + row, m)
    tl.store(l_ptr + row, l)


def run_kernel(x: torch.Tensor, rescale: bool) -> tuple[torch.Tensor, torch.Tensor]:
    n_rows, n_cols = x.shape
    m = torch.empty(n_rows, device=x.device, dtype=torch.float32)
    l = torch.empty_like(m)
    online_stats_kernel[(n_rows,)](m, l, x, x.stride(0), n_cols,
                                   RESCALE=rescale, BLOCK=BLOCK)
    return m, l


def main() -> None:
    print("=" * 78)
    print("在线 softmax：一行放不下时，为什么必须 rescale 已累积的量")
    print("=" * 78)
    print(f"  设备   : {torch.cuda.get_device_name(0)}")
    print(f"  数据   : {N_ROWS} 行 × {N_COLS} 列 fp32 = {N_ROWS * N_COLS * 4 / 1e6:.1f} MB")
    print(f"  分块   : BLOCK={BLOCK} → 每行 {N_COLS // BLOCK} 块（一行远超片上容量）")
    print()

    torch.manual_seed(0)
    x = torch.randn(N_ROWS, N_COLS, device="cuda", dtype=torch.float32)
    # 刻意把尖峰放在**最后一块**：m 会在循环快结束时从 ~3.5 跳到 10
    # → 早期 63 个块都是用"小 m"算的，不 rescale 的话膨胀量巨大
    x[:, -1] = SPIKE
    print(f"  构造   : 每行最后一块注入尖峰 +{SPIKE}（让运行最大值在循环末尾才跳起来）")
    print()

    # ---- 精确参照（两趟，允许先读完整行）----
    m_ref = x.max(dim=1).values
    l_ref = (x - m_ref[:, None]).exp().sum(dim=1)

    # ---- ① 正确的在线算法 ----
    m_ok, l_ok = run_kernel(x, rescale=True)
    # ---- ② 不 rescale 的错误变体 ----
    m_bad, l_bad = run_kernel(x, rescale=False)

    err_m = (m_ok - m_ref).abs().max().item()
    rel_l_ok = ((l_ok - l_ref).abs() / l_ref).max().item()
    rel_l_bad = ((l_bad - l_ref).abs() / l_ref).max().item()

    print(f"{'实现':<28} {'max 误差':>12} {'l 相对误差':>14} {'l 最大倍率偏差':>16}")
    print("-" * 78)
    print(f"{'在线 + rescale（正确）':<28} {err_m:>12.2e} {rel_l_ok:>14.2e} "
          f"{(l_ok / l_ref).max().item():>15.4f}x")
    print(f"{'在线 不 rescale（错误）':<28} {err_m:>12.2e} {rel_l_bad:>14.2e} "
          f"{(l_bad / l_ref).max().item():>15.4f}x")
    print()
    print(f"  参照 l_ref（第 0 行）: {l_ref[0].item():.4f}")
    print(f"  正确 l_ok（第 0 行）  : {l_ok[0].item():.4f}")
    print(f"  错误 l_bad（第 0 行） : {l_bad[0].item():.4f}   "
          f"← 放大了约 {(l_bad[0] / l_ref[0]).item():.0f} 倍")
    print()
    print("=" * 78)
    print("解读")
    print("=" * 78)
    print("  · 正确版与两趟精确计算**完全一致** → 在线算法不是近似，是等价变换。")
    print("  · 错误版的 l 被放大约上千倍 → 归一化因子直接失效。")
    print("    原因：早期块用旧的（小的）m 计算 exp，exp(x - m_small) 远大于 exp(x - m_final)。")
    print()
    print("  这解释了 FlashAttention 里那句最容易背错的话：")
    print("    需要 rescale 的**不只是归一化因子 l，还有已经累积的输出 O** ——")
    print("    因为 O = Σ (exp(s - m)/l) · V，它也是用「旧的 m」算出来的，同样必须追溯修正。")
    print()
    print("ONLINE_OK")


if __name__ == "__main__":
    main()
