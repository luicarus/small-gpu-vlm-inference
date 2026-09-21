"""实测 GPU 真实可用显存与算力（区分「读数占用」与「实际占用」）。

背景：nvidia-smi 报 3359/4096 MiB 已用、且无任何进程可见，WSL 重启也没回收。
需要确认：这到底是「驱动真占用了显存」还是「WDDM 读数虚高但实际仍能分配」。
"""
from __future__ import annotations

import time

import torch

print(f"GPU: {torch.cuda.get_device_name(0)}")
free, total = torch.cuda.mem_get_info()
print(f"torch 报告: 空闲 {free / 1024**3:.2f} GiB / 总计 {total / 1024**3:.2f} GiB")
print()

print("=== 逐级分配测试（找真实可用上限）===")
blocks = []
step_mib = 128
try:
    while True:
        blocks.append(torch.empty(step_mib * 1024 * 1024, dtype=torch.uint8, device="cuda"))
except torch.cuda.OutOfMemoryError:
    pass
print(f"  ✅ 实际可分配上限 ≈ {len(blocks) * step_mib} MiB")
del blocks
torch.cuda.empty_cache()

print()
print("=== 算力实测（1024³ bf16 matmul × 50 次）===")
a = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
b = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
for _ in range(5):
    a @ b
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    a @ b
torch.cuda.synchronize()
dt = time.perf_counter() - t0
flops = 50 * 2 * 1024**3
print(f"  耗时 {dt * 1000:.0f} ms → {flops / dt / 1e12:.2f} TFLOPS")
print("  （3050 Ti 移动版 bf16 峰值约 17-35 TFLOPS；远低于 1 说明卡不可用）")

free2, _ = torch.cuda.mem_get_info()
print()
print(f"释放后空闲: {free2 / 1024**3:.2f} GiB")
print("GPU_PROBE_DONE")