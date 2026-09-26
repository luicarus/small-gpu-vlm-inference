"""vector add 的 N 扫描 —— 验证「小负载测的是启动开销，不是带宽」。

## 为什么要做这一步

`check_triton.py` 在 n=98432（约 1.18 MB 流量）上测出 torch 148 GB/s vs triton 81 GB/s，
但 **两个数字都远低于这张卡 ~192 GB/s 的理论带宽**，且 8 µs 的耗时已经贴着
「1.18 MB ÷ 192 GB/s = 6.2 µs」的理论地板 —— 说明那次测量被**内核启动开销**主导，
而不是带宽。

**如果不做这个扫描，会得出一个错误的结论**：「Triton 比 torch 慢」。
正确结论应该是：「在小负载上 Triton 的启动开销更高；负载变大后固定开销被摊薄」。

## 怎么验证

把 N 从 2^14 扫到 2^26（流量 192 KB → 805 MB），观察：
  - 小 N：两者都远低于峰值，且差距大 → 启动开销主导
  - 大 N：两者都逼近理论带宽，**差距收敛** → 带宽主导

同时用「两负载差分」粗估启动开销：
  t(N) ≈ overhead + bytes(N) / 有效带宽
取足够大的两个点做线性拟合，截距就是每 launch 的固定开销量级。

## 输出什么

每个 N 一行：耗时 / 实测带宽 / 占理论峰值百分比 / triton 相对 torch 的倍率。
最后给出拟合出的固定开销估计。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import triton

# 复用 check_triton.py 里的 add_kernel —— 同目录，避免两份 kernel 定义漂移
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_triton import add_kernel  # noqa: E402

PEAK_BW_GBS = 192.0   # RTX 3050 Ti Laptop 理论带宽（128-bit GDDR6 @ 12 Gbps）
BLOCK = 1024

# (n, iters)：小负载多跑几次压噪声，大负载少跑几次控总时长
SIZES = [
    (2**14, 300),
    (2**16, 300),
    (2**18, 200),
    (2**20, 200),
    (2**22, 100),
    (2**24, 50),
    (2**26, 20),
]


def bench(fn, warmup: int = 20, iters: int = 200) -> float:
    """返回单次调用平均毫秒数。

    warmup 给 20 次而不是 1 次：check_triton.py 里实测「编译完成后的第 2 次调用」
    仍比稳态慢 9 倍（launcher 元数据 / cubin 加载等冷路径），只 warm 1 次会系统性偏高。
    """
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
    print("vector add · N 扫描：小负载测的是启动开销，大负载才是带宽")
    print("=" * 78)
    print(f"  设备      : {torch.cuda.get_device_name(0)}")
    print(f"  理论带宽  : {PEAK_BW_GBS:.0f} GB/s（实际可达通常 80~90%）")
    print(f"  BLOCK_SIZE: {BLOCK}")
    print()
    print(f"{'n':>10} {'流量MB':>8} | {'triton ms':>9} {'GB/s':>7} {'占峰值':>6} | "
          f"{'torch ms':>9} {'GB/s':>7} {'占峰值':>6} | {'torch/triton':>12}")
    print("-" * 78)

    rows = []
    for n, iters in SIZES:
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        grid = (triton.cdiv(n, BLOCK),)
        traffic_mb = n * 4 * 3 / 1e6      # 读 x + 读 y + 写 out

        t_tri = bench(lambda: add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK), iters=iters)
        t_tor = bench(lambda: torch.add(x, y, out=out), iters=iters)

        bw_tri = traffic_mb / t_tri          # MB/ms == GB/s
        bw_tor = traffic_mb / t_tor
        rows.append((n, traffic_mb, t_tri, bw_tri, t_tor, bw_tor))

        print(f"{n:>10} {traffic_mb:>8.1f} | {t_tri:>9.4f} {bw_tri:>7.1f} "
              f"{bw_tri / PEAK_BW_GBS * 100:>5.0f}% | {t_tor:>9.4f} {bw_tor:>7.1f} "
              f"{bw_tor / PEAK_BW_GBS * 100:>5.0f}% | {t_tor / t_tri:>12.2f}")

        del x, y, out
        torch.cuda.empty_cache()

    # ---- 用最小与最大两个负载差分，粗估每 launch 的固定开销 ----
    # t = overhead + traffic / BW  →  overhead = (t2 - t1) 中无法被带宽解释的部分
    n_small, mb_small, t_tri_s, bw_tri_s, t_tor_s, bw_tor_s = rows[0]
    n_big, mb_big, t_tri_b, bw_tri_b, t_tor_b, bw_tor_b = rows[-1]

    print()
    print("=" * 78)
    print("解读")
    print("=" * 78)
    print(f"  最大负载 {mb_big:.0f} MB 上：triton {bw_tri_b:.0f} GB/s "
          f"({bw_tri_b / PEAK_BW_GBS * 100:.0f}% 峰值) vs torch {bw_tor_b:.0f} GB/s "
          f"({bw_tor_b / PEAK_BW_GBS * 100:.0f}% 峰值)")
    print(f"  倍率从最小负载的 {t_tor_s / t_tri_s:.2f}× 收敛到最大负载的 {t_tor_b / t_tri_b:.2f}×")
    print()
    print("  若倍率随负载增大而收敛 → 证实「小负载差距 = 启动开销」，而不是 Triton 慢")
    print("  若大负载下仍差很多    → 说明 Triton 的 kernel 本身写得不够好（尾部 mask、向量化）")
    print()
    print(f"  固定开销粗估（用 traffic→time 的截距思路）：")
    print(f"    triton 在大负载下的有效带宽 {bw_tri_b:.0f} GB/s 折算 {mb_big / bw_tri_b:.4f} ms，"
          f"实测 {t_tri_b:.4f} ms → 差 {t_tri_b - mb_big / bw_tri_b:.4f} ms")
    print(f"    torch  在大负载下的有效带宽 {bw_tor_b:.0f} GB/s 折算 {mb_big / bw_tor_b:.4f} ms，"
          f"实测 {t_tor_b:.4f} ms → 差 {t_tor_b - mb_big / bw_tor_b:.4f} ms")
    print()
    print("BENCH_SWEEP_OK")


if __name__ == "__main__":
    main()
