"""诊断：搞清 torch.profiler 的 device_time_total 单位，以及 kernel 实际耗时占比。

为什么单独写这个：v2 输出里 GEMM 的 "us_per_call" 与墙钟对不上（差 25 倍），
说明单位换算错了。用「总 device 时间 / 迭代数 / 墙钟单次耗时」三者对照就能定出真实单位。
"""
from __future__ import annotations

import torch
from torch.profiler import ProfilerActivity, profile

import bitsandbytes as bnb

DTYPE = torch.bfloat16
K, N, M = 3456, 1280, 512
ITERS = 30

x = torch.randn(M, K, dtype=DTYPE, device="cuda")
q = bnb.nn.Linear4bit(K, N, bias=False, quant_type="nf4", compute_dtype=DTYPE).cuda()
q(x); torch.cuda.synchronize()
for _ in range(10):
    q(x)
torch.cuda.synchronize()

s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(ITERS):
    q(x)
e.record(); torch.cuda.synchronize()
wall_total_ms = s.elapsed_time(e)
wall_per_call_ms = wall_total_ms / ITERS

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(ITERS):
        q(x)
    torch.cuda.synchronize()

print(f"形状: M={M} K={K} N={N}  迭代 {ITERS} 次")
print(f"墙钟: 总 {wall_total_ms:.3f} ms  → 单次 {wall_per_call_ms * 1000:.1f} µs")
print()
print(f"{'kernel':<50} {'count':>7} {'dev_total':>12} {'self_dev_total':>14}")
print("-" * 88)
sum_dev = 0.0
for ev in prof.key_averages():
    if ev.device_time_total <= 0:
        continue
    sum_dev += ev.device_time_total
    print(f"{ev.key[:48]:<50} {ev.count:>7} {ev.device_time_total:>12.0f} {ev.self_device_time_total:>14.0f}")

print("-" * 88)
print(f"{'所有 kernel 的 device_time_total 之和':<50} {'':>7} {sum_dev:>12.0f}")
print()
print("=== 单位推断 ===")
print(f"若 device_time_total 单位是 µs: 总 GPU 时间 = {sum_dev / 1000:.2f} ms, "
      f"单次 = {sum_dev / ITERS / 1000:.3f} ms  (墙钟单次 {wall_per_call_ms:.3f} ms)")
print(f"若单位是 ns:                    总 GPU 时间 = {sum_dev / 1e6:.2f} ms, "
      f"单次 = {sum_dev / ITERS / 1e6:.3f} ms")
print()
print("=== 结论：看哪一个与墙钟单次耗时吻合，那个就是正确单位 ===")
