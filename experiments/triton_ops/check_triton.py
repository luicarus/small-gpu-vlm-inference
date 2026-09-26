"""Triton 可用性验证 —— 最小 vector add kernel。

## 为什么要先做这一步
Triton 已经作为 vLLM/SGLang 的依赖装上了（3.7.1），所以"安装"不需要额外动作。
但**依赖存在 ≠ 能独立编译运行**：
  - Triton 自带 LLVM 后端，不依赖系统 nvcc（本机 nvcc 是 CUDA 12.0，很旧）；
  - 但仍要确认它能在这个 sm86 设备上用当前 driver（610.47）编译并通过 ptxas。
这一步就是把这三点钉死，后面写 RMSNorm 才有可信的基线。

## 输出什么
1. 版本与设备信息；
2. 编译 + 运行一个 vector add，验证数值正确；
3. 首次编译耗时（后面写算子时要拿它当"warmup 成本"的量级参考）。
"""

from __future__ import annotations

import time

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,          # *fp32  输入 x
    y_ptr,          # *fp32  输入 y
    out_ptr,        # *fp32  输出
    n_elements,     # int    元素总数
    BLOCK_SIZE: tl.constexpr,   # 每个 program 处理多少元素（编译期常量）
):
    """经典的 grid-stride 写法：每个 program 负责 BLOCK_SIZE 个元素。

    Triton 的心智模型是「一个 program 处理一块数据」：
      - `tl.program_id(0)` 相当于 CUDA 的 blockIdx.x；
      - `tl.arange` 生成块内下标，offset 决定这块数据在全局的位置；
      - mask 处理尾部越界 —— Triton 里**必须显式**做，不像 CUDA 那样可以只靠边界判断。
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def main() -> None:
    print("=" * 62)
    print("Triton 可用性验证")
    print("=" * 62)
    print(f"  triton     : {triton.__version__}")
    print(f"  torch      : {torch.__version__}")
    print(f"  device     : {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability()
    print(f"  capability : sm{cap[0]}{cap[1]}")
    print()

    n = 98432                       # 故意取非 2 的幂，逼出 mask 分支
    BLOCK = 1024
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    grid = (triton.cdiv(n, BLOCK),)
    print(f"  n={n}  BLOCK_SIZE={BLOCK}  grid={grid}")

    # ---- 第一次：包含编译开销 ----
    t0 = time.perf_counter()
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK)
    torch.cuda.synchronize()
    t_compile = time.perf_counter() - t0

    # ---- 第二次：应走缓存 ----
    t0 = time.perf_counter()
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK)
    torch.cuda.synchronize()
    t_warm = time.perf_counter() - t0

    ref = x + y
    ok = torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
    max_err = (out - ref).abs().max().item()

    print()
    print(f"  首次（含编译）: {t_compile * 1000:8.2f} ms")
    print(f"  二次（走缓存）  : {t_warm * 1000:8.2f} ms")
    print(f"  数值正确        : {'✅' if ok else '❌'}  (最大误差 {max_err:.2e})")
    print()

    if not ok:
        raise SystemExit("数值校验失败")

    # ---- 与 torch 原生对比一下带宽（顺带看看这个规模下的量级）----
    def bench(fn, iters=200):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    t_triton = bench(lambda: add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK))
    t_torch = bench(lambda: torch.add(x, y, out=out))
    bytes_moved = n * 4 * 3          # 读 x + 读 y + 写 out
    print("  同规模下（vector add）")
    print(f"    triton : {t_triton:7.4f} ms  → {bytes_moved / (t_triton/1000) / 1e9:7.1f} GB/s")
    print(f"    torch  : {t_torch:7.4f} ms  → {bytes_moved / (t_torch/1000) / 1e9:7.1f} GB/s")
    print()
    print("TRITON_OK")


if __name__ == "__main__":
    main()
