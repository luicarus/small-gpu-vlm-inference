"""FlashAttention 骨架 —— 把「显存从平方降到线性」测出来。

## 这个脚本要回答的问题

    「能讲清 FlashAttention 为什么把显存从平方降到线性」

不讲结论、不讲论文原话，**用这条 4GB 卡上的峰值显存实测**回答：

    朴素注意力：S = Q@K^T 必须**物化**成 N×N 矩阵 → 显存 O(N²)
    FlashAttention：S 只存在于寄存器/SRAM，**从不落盘** → 显存 O(N)

## 与同目录其他脚本的衔接

    softmax_tutorial.py   → 融合：中间结果不落盘（省 IO）
    online_softmax_toy.py → 一行放不下时，用运行 max + rescale 边算边修正
    flash_attention_toy.py（本脚本）→ 把两者套进 attention 的循环：
                              S = Q@K^T   ← 这一步在朴素实现里要落盘 N×N
                              P = exp(S - m)  ← 在线 softmax
                              O += P@V        ← 累加器，**并且要 rescale**

## FA 为什么能在线累积 O（关键）

    O = P @ V  中，P 是 (M × block_N)、V 是 (block_N × d)
    → 输出形状 (M × d)，**block_N 被消掉** —— 累加器尺寸与循环次数无关。

    对比普通 softmax：输出就是被归约的那一维（N），所以放不下、必须两趟。

## 输出什么

1. **正确性**：FA 输出与朴素实现逐元素对比（fp16 容差内一致）；
2. **峰值显存扫描**：N 从 512 到 16384，朴素 vs FA 的峰值显存与耗时
   → 朴素的 N² 增长 vs FA 的平缓，就是本实验要给出的答案。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent))

D_HEAD = 64          # head_dim：FA 的累加器宽度就是这个数（与序列长度无关）
BLOCK_M = 64
BLOCK_N = 64
SIZES = [512, 1024, 2048, 4096, 8192, 16384]


@triton.jit
def flash_attn_kernel(
    Q, K, V, Out,
    sm_scale,
    N,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """单头、非因果的 FlashAttention 前向。

    ★ 对照 FlashAttention 论文算法 1 逐行看：
        m_i → 运行最大值 m
        l_i → 运行归一化因子 l
        acc → 输出累加器 O
      「在线」体现在：每读一个 KV 块，m 可能变大 → l 与 acc 都乘 alpha 追溯修正。
    """
    pid_m = tl.program_id(axis=0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q = tl.load(Q + offs_m[:, None] * D + offs_d[None, :])      # (M, D)

    # ---- 三个在线状态量：全部驻留寄存器，与 KV 块数量无关 ----
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)   # 运行 max
    l_i = tl.zeros([BLOCK_M], tl.float32)                 # 运行 sum(exp)
    acc = tl.zeros([BLOCK_M, D], tl.float32)              # ★ 输出累加器 O（M×D，恒定）

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k = tl.load(K + offs_n[:, None] * D + offs_d[None, :])   # (BN, D)
        # ★ S = Q@K^T 只在寄存器里存在 —— 朴素实现会把它物化成 N×N 落盘
        s = tl.dot(q, tl.trans(k)) * sm_scale                    # (M, BN)

        m_new = tl.maximum(m_i, tl.max(s, axis=1))               # 运行 max 更新
        p = tl.exp(s - m_new[:, None])                           # ★ exp(S - m)，不落盘
        alpha = tl.exp(m_i - m_new)                              # ★★★ rescale 因子 ★★★

        l_i = l_i * alpha + tl.sum(p, axis=1)                    # 归一化因子追溯修正
        acc = acc * alpha[:, None]                               # ★ 输出累加器也要修正！

        v = tl.load(V + offs_n[:, None] * D + offs_d[None, :])   # (BN, D)
        # ⚠️ Triton 3.7 起 tl.dtype 没有 element_ty 属性（旧教程的 p.to(v.dtype.element_ty) 会报错），
        #    直接传 v.dtype 即可；tl.dot 要求两侧 dtype 一致，所以 p 必须先转成 v 的 dtype。
        acc = tl.dot(p.to(v.dtype), v, acc)                      # ★ acc += P@V（累加进 acc）

        m_i = m_new

    acc = acc / l_i[:, None]                                     # 最后一次性归一化
    tl.store(Out + offs_m[:, None] * D + offs_d[None, :],
             acc.to(tl.float16))


def flash_attention(q, k, v, sm_scale):
    n, d = q.shape
    out = torch.empty_like(q)
    grid = (triton.cdiv(n, BLOCK_M),)
    flash_attn_kernel[grid](q, k, v, out, sm_scale, n, D=d,
                            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out


def naive_attention(q, k, v, sm_scale):
    """朴素实现：S 必须物化成 N×N —— 显存 O(N²) 的来源。"""
    s = (q @ k.transpose(-1, -2)) * sm_scale      # ← N×N 矩阵在这里被创建
    p = torch.softmax(s, dim=-1)                  # ← 又一个 N×N 中间矩阵
    return p @ v


def measure(fn, *args, iters: int = 5):
    """返回 (峰值显存 MB, 单次耗时 ms)；OOM 时返回 (None, None)。"""
    try:
        for _ in range(2):                        # warmup（含 JIT 编译）
            fn(*args)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(*args)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1000
        peak = torch.cuda.max_memory_allocated() / 1024**2
        return peak, ms
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None


def main() -> None:
    print("=" * 84)
    print("FlashAttention 骨架：朴素 vs FA 的峰值显存扫描")
    print("=" * 84)
    print(f"  设备      : {torch.cuda.get_device_name(0)}")
    print(f"  head_dim  : {D_HEAD}（= FA 累加器宽度，与序列长度无关）")
    print(f"  BLOCK_M/N : {BLOCK_M} / {BLOCK_N}")
    print(f"  dtype     : fp16")
    print()

    sm_scale = 1.0 / (D_HEAD ** 0.5)

    # ---------- ① 正确性（小规模）----------
    torch.manual_seed(0)
    n = 512
    q = torch.randn(n, D_HEAD, device="cuda", dtype=torch.float16)
    k = torch.randn(n, D_HEAD, device="cuda", dtype=torch.float16)
    v = torch.randn(n, D_HEAD, device="cuda", dtype=torch.float16)

    o_fa = flash_attention(q, k, v, sm_scale)
    o_ref = naive_attention(q, k, v, sm_scale)
    diff = (o_fa.float() - o_ref.float()).abs()
    rel = diff.max().item() / o_ref.float().abs().max().item()

    print("① 正确性校验（N=512, D=64, fp16）")
    print(f"   最大绝对误差: {diff.max().item():.3e}")
    print(f"   最大相对误差: {rel:.3e}   "
          f"{'✅ 通过' if rel < 1e-2 else '❌ 不通过（fp16 容差 1e-2）'}")
    print()

    # ---------- ② 峰值显存扫描 ----------
    print("② 峰值显存 / 耗时扫描（fp16, D=64，非因果）")
    print()
    print(f"{'N':>7} | {'朴素峰值MB':>11} {'朴素 ms':>9} | "
          f"{'FA 峰值MB':>10} {'FA ms':>8} | {'显存比':>7} {'加速':>7}")
    print("-" * 84)

    results = []
    for n in SIZES:
        torch.cuda.empty_cache()
        q = torch.randn(n, D_HEAD, device="cuda", dtype=torch.float16)
        k = torch.randn(n, D_HEAD, device="cuda", dtype=torch.float16)
        v = torch.randn(n, D_HEAD, device="cuda", dtype=torch.float16)

        peak_naive, ms_naive = measure(naive_attention, q, k, v, sm_scale)
        peak_fa, ms_fa = measure(flash_attention, q, k, v, sm_scale)

        naive_s = f"{peak_naive:>11.2f}" if peak_naive else f"{'OOM':>11}"
        naive_t = f"{ms_naive:>9.2f}" if ms_naive else f"{'—':>9}"
        fa_s = f"{peak_fa:>10.2f}" if peak_fa else f"{'OOM':>10}"
        fa_t = f"{ms_fa:>8.2f}" if ms_fa else f"{'—':>8}"
        ratio = f"{peak_naive / peak_fa:>7.1f}x" if (peak_naive and peak_fa) else f"{'—':>7}"
        speed = f"{ms_naive / ms_fa:>7.2f}x" if (ms_naive and ms_fa) else f"{'—':>7}"
        print(f"{n:>7} | {naive_s} {naive_t} | {fa_s} {fa_t} | {ratio} {speed}")
        results.append((n, peak_naive, peak_fa))
        del q, k, v
        torch.cuda.empty_cache()

    print()
    print("=" * 84)
    print("读表（本实验的结论）")
    print("=" * 84)
    print("  · 朴素实现的峰值显存随 N **平方增长**（S 与 softmax 各物化一个 N×N 矩阵）；")
    print("  · FA 的峰值显存随 N **近似线性**（只有 Q/K/V/O，都是 N×d）；")
    print("  · 两者差值就是「不落盘 N×N 分数矩阵」省下的显存。")
    print()
    print("  为什么 FA 能做到？三个条件缺一不可：")
    print("    ① 分块：S 只在寄存器/SRAM 里存在，从不写回显存（融合思想，见 softmax_tutorial.py）")
    print("    ② 在线：max 边算边更新，l 与 O 用 alpha=exp(m_old-m_new) 追溯修正（见 online_softmax_toy.py）")
    print("    ③ 输出维度 = head_dim，与循环次数无关 → 累加器尺寸恒定，所以能在线累积")
    print()
    print("  一句话：**FA 把「物化 N×N」换成了「循环 N/BLOCK_N 次、每次只碰 BLOCK_N×d」**，")
    print("  显存从 O(N²) 降到 O(N·d)。**代价是重算**（不做反向所需的中间量保存）——这是另一个话题。")
    print()
    print("FLASH_ATTN_OK")


if __name__ == "__main__":
    main()
