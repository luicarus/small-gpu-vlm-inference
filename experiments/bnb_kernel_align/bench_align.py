"""测量 bnb 4bit GEMM 的快慢路径差异（v2：用 profiler 直接判定走的是哪条路）。

## 为什么要写 v2

v1 靠「有没有打印对齐警告」来判断是否走了慢路径 —— **这是错的**。
源码 `bitsandbytes/backends/cuda/ops.py:934` 的判定是：

    if M > _gemm_4bit_custom_max_m:      # CUDA = 1536
        use_custom = False               # ← 静默回退，无警告
    elif K % blocksize != 0:
        warn("inner dimension ... not aligned ...")   # ← 只有这条会警告
        use_custom = False
    else:
        use_custom = _gemm_4bit_use_custom_fn(device_index, dtype, M, N, K)
        # ↑ 这个「按架构逐代校准」的启发式也可能返回 False，同样静默

所以「无警告」≠「走快路径」。本版改用 torch.profiler 抓真实 kernel 名：
  * 自定义融合 kernel → `cgemm_4bit_*`
  * 回退路径          → `dequantize_4bit`（反量化）+ 普通 GEMM（gemm/cutlass/cublas）

这样「走哪条路」和「耗时多少」都由同一份 profiler 数据给出，不再靠推断。

## 被测形状的含义

真实模型里触发警告的是 Qwen2.5-VL 的 ViT 降维层：权重 [1280, 3420]，
对应 GEMM 的 N=1280、K=3420（3420 % 64 = 28 → 不对齐）。
本脚本固定 N=1280，扫描 K（对齐与否）与 M（token 数），并记录实测的 SM 数与启发式结论。
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import torch


def bench_with_profiler(layer, x, iters: int = 30):
    """返回 (单次调用耗时 ms, kernel 统计 dict, 是否发出对齐警告)。

    用 profiler 抓 CUDA kernel 级的真实耗时，并据此判定走了哪条路径——
    而不是靠"有没有警告"这种间接推断。
    """
    from torch.profiler import ProfilerActivity, profile

    layer(x)
    torch.cuda.synchronize()
    for _ in range(10):          # warmup
        layer(x)
    torch.cuda.synchronize()

    # --- 墙钟计时（5 轮取中位）---
    times = []
    for _ in range(5):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            layer(x)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / iters)
    times.sort()
    wall_ms = times[len(times) // 2]

    # --- profiler 抓 kernel ---
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                layer(x)
            torch.cuda.synchronize()
    warned = any("not aligned for fast kernel" in str(w.message) for w in caught)

    kernels = {}
    for ev in prof.key_averages():
        if ev.device_time_total > 0:
            # 【单位】device_time_total 单位是**微秒**（实测校准：同形状下所有 kernel 之和
            # 10.30 ms vs 墙钟 10.76 ms，占 96%）。除以该 kernel 的发射次数即得单次耗时。
            # v2 曾误乘 1000 导致数值虚高三个数量级，故此处显式注释单位。
            per_call_us = ev.device_time_total / max(ev.count, 1)
            kernels[ev.key] = {
                "count_per_call": round(ev.count / iters, 2),
                "us_per_call": round(per_call_us, 2),
                "us_total_in_profile": round(ev.device_time_total, 1),
            }
    return wall_ms, kernels, warned


def classify(kernels: dict) -> tuple[str, str]:
    """按 kernel 名判定路径。

    注意 kernel 命名有两套：
      * bnb 自定义融合 kernel：`gemm_4bit_sm80_m16n8k16`（C 层实现，名字不带 c 前缀）
      * 回退路径：`kDequantizeBlockwise`（反量化）+ cuBLAS/cutlass 的普通 GEMM
    """
    names = " ".join(kernels.keys()).lower()
    has_custom = "gemm_4bit" in names
    has_dequant = "dequantize" in names or "dequant" in names
    if has_custom and not has_dequant:
        return "custom", "bnb 自定义融合 kernel"
    if has_dequant and not has_custom:
        return "fallback", "反量化 + 普通 GEMM"
    if has_custom and has_dequant:
        return "mixed", "两者都有"
    return "unknown", "未识别"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3,
                    help="每个配置独立重复测几次，用于区分真实差异与噪声（默认 3）")
    args = ap.parse_args()

    import bitsandbytes as bnb

    dtype = getattr(torch, args.dtype)
    props = torch.cuda.get_device_properties(0)
    num_sms = props.multi_processor_count
    major, minor = props.major, props.minor
    print(f"GPU: {props.name}  sm_{major}{minor}  SMs={num_sms}  dtype={args.dtype}")

    N = 1280                                   # ViT fc2 的输出维
    n_blocks = (N + 63) // 64                  # 启发式里的 wave 计算方式
    print(f"形状: N={N} → n_blocks={n_blocks}（启发式用 n_blocks 与 SM 数比较）")
    print(f"启发式预判（sm86 分支）: n_blocks>=SMs → 仅 M<={128} 用自定义 kernel")
    print()

    # K=3420 真实值（不对齐）；3456/3328/2048 对齐
    configs = [(3420, 128), (3456, 128), (3420, 512), (3456, 512),
               (3420, 1024), (3456, 1024), (3456, 2048)]

    records = []
    print(f"{'K':>5} {'mod64':>5} {'M':>5} | {'路径':<9} {'警告':<4} | "
          f"{'bf16 ms':>8} {'nf4 ms':>8} | {'nv4快':>6} | profiler 主 kernel")
    print("-" * 108)

    for k, m in configs:
        x = torch.randn(m, k, dtype=dtype, device="cuda")

        ref = torch.nn.Linear(k, N, bias=False, dtype=dtype).cuda()
        t_ref = bench_with_profiler(ref, x, args.iters)[0]

        # 【为什么要 repeats】CUDA 计时本身有抖动；单次测量下「1.3× 差异」无法区分
        # 真实效应与噪声。重复 N 次取中位，并记录极差，才能判断差异是否可信。
        q = bnb.nn.Linear4bit(k, N, bias=False,
                              quant_type="nf4", compute_dtype=dtype).cuda()
        samples, kernels, warned = [], {}, False
        for _ in range(args.repeats):
            t, kern, w = bench_with_profiler(q, x, args.iters)
            samples.append(t)
            if not kernels:
                kernels = kern
            warned = warned or w
        samples.sort()
        t_q = samples[len(samples) // 2]
        spread = samples[-1] - samples[0]

        path, path_cn = classify(kernels)
        # 挑出耗时最高的 kernel 名（去掉 ::memory 之类的辅助项）
        top = sorted(((v["us_per_call"], n) for n, v in kernels.items()), reverse=True)[:2]
        top_s = ", ".join(f"{n.split('(')[0][:34]}({us:.0f}µs)" for us, n in top)

        print(f"{k:>5} {k % 64:>5} {m:>5} | {path:<9} {str(warned):<4} | "
              f"{t_ref:>8.4f} {t_q:>8.4f} ±{spread:.4f} | {t_ref / t_q:>5.2f}× | {top_s}")

        records.append({
            "K": k, "N": N, "M": m, "K_mod64": k % 64,
            "path": path, "warned": warned,
            "bf16_ms": round(t_ref, 4), "nf4_ms": round(t_q, 4),
            "nf4_samples_ms": [round(s, 4) for s in samples],
            "nf4_spread_ms": round(spread, 4),
            "speedup_vs_bf16": round(t_ref / t_q, 3),
            "kernels": kernels,
            "gpu": props.name, "sm": f"{major}{minor}", "num_sms": num_sms,
        })

        del ref, q, x
        torch.cuda.empty_cache()

    print()
    print("=== 只看 nf4：同 M 下 K 对齐 vs 不对齐 ===")
    print(f"{'M':>5} | {'K=3456 对齐':>12} | {'K=3420 不对齐':>14} | {'倍率':>7} | 对齐走的路 | 不对齐走的路")
    print("-" * 86)
    for m in (128, 512, 1024):
        a = next(r for r in records if r["K"] == 3456 and r["M"] == m)
        b = next(r for r in records if r["K"] == 3420 and r["M"] == m)
        print(f"{m:>5} | {a['nf4_ms']:>9.4f} ms | {b['nf4_ms']:>11.4f} ms | "
              f"{b['nf4_ms'] / a['nf4_ms']:>6.2f}× | {a['path']:<9} | {b['path']}")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fp:
            for r in records:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n结果已写入 {p}")


if __name__ == "__main__":
    main()
