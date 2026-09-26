"""SGLang 冒烟：能不能在 4GB 卡上跑 Qwen2-VL-2B-AWQ。

## 为什么要先冒烟
三组已验证实验都用 vLLM。SGLang 是**另一套引擎**，显存模型不同
（vLLM 用 `gpu_memory_utilization` 划预算，SGLang 用 `mem_fraction_static` 划静态池），
在 4GB 上能不能起来、装不装得下权重 + KV，必须先探清楚再谈对比。

## 要回答的问题
1. 能否加载 2B AWQ 权重（2.74 GiB）并启动？
2. 启动后 KV 池有多少？（决定能跑多长序列、多少并发）
3. **prefix caching / RadixAttention 命中情况**——SGLang 默认开启，能不能拿到命中指标？
4. 单图推理能否正常工作（视觉塔是否被正确加载）？

## 与主实验的关系
成功后，才把 exp3 的共享前缀 workload 移植过来做对比（见同目录 `run_frames.sh`）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# 复用主实验的路径锚点与素材（不写死绝对路径：脚本搬迁后会静默指错）
BENCH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BENCH / "experiments" / "vlm_inference_benchmark"))
from shared.common import DEFAULT_IMAGE, MODEL_AWQ  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_AWQ)
    ap.add_argument("--mem-fraction-static", type=float, default=0.80,
                    help="SGLang 的静态显存比例。实测 0.75 时权重加载后无 KV 空间"
                         "（SGLang 自己算出最小可用 0.7508），取 0.80 与 vLLM 侧一致")
    ap.add_argument("--quantization", default="awq_marlin",
                    help="SGLang 检测到该模型可用 awq_marlin（更快的 kernel）；"
                         "显式写 awq 会强制走未优化的后端，日志会专门警告")
    ap.add_argument("--attention-backend", default="triton",
                    help="默认 flashinfer 会 JIT 编译 attention kernel，而它生成的编译命令"
                         "带 --compress-mode=size（CUDA 12.1+ 才支持），本机系统 nvcc 是 12.0；"
                         "改用 venv 的 CUDA 13.4 又会与 flashinfer 自带的 CCCL 头文件版本冲突。"
                         "triton 后端不依赖 flashinfer，可绕开这组版本矛盾。"
                         "（fa3/fa4 分别是 Hopper/Blackwell 的，sm86 用不了）")
    ap.add_argument("--context-length", type=int, default=2048)
    ap.add_argument("--max-total-tokens", type=int, default=0, help="0 = 交给 SGLang 推导")
    ap.add_argument("--image", default=str(DEFAULT_IMAGE))
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--cuda-graph-max-bs-decode", type=int, default=0,
                    help="0 = 用 SGLang 默认（实测捕获 bs=[1,2,4,8] 耗时 337s）。"
                         "调小可加快启动、并给 KV 池腾预算。"
                         "注意参数名是 cuda_graph_max_bs_decode（对应 CLI --cuda-graph-max-bs-decode），"
                         "写成 cuda_graph_max_bs 会 TypeError")
    ap.add_argument("--disable-cuda-graph", action="store_true",
                    help="完全关掉 CUDA graph 捕获（启动最快，但推理会慢；仅用于快速验证）")
    ap.add_argument("--question", default="请用一句话描述这张图的内容。")
    args = ap.parse_args()

    from PIL import Image
    import sglang as sgl

    print(f"模型        : {args.model}")
    print(f"量化        : {args.quantization}")
    print(f"attn backend: {args.attention_backend}")
    print(f"mem_fraction: {args.mem_fraction_static}")
    print(f"context_len : {args.context_length}")
    print(f"图片        : {args.image}")
    print()

    image = Image.open(args.image).convert("RGB")

    kwargs = dict(
        model_path=args.model,
        quantization=args.quantization,
        attention_backend=args.attention_backend,
        mem_fraction_static=args.mem_fraction_static,
        context_length=args.context_length,
        trust_remote_code=True,
        log_level="info",
    )
    if args.max_total_tokens > 0:
        kwargs["max_total_num_tokens"] = args.max_total_tokens
    if args.cuda_graph_max_bs_decode > 0:
        kwargs["cuda_graph_max_bs_decode"] = args.cuda_graph_max_bs_decode
    if args.disable_cuda_graph:
        kwargs["disable_cuda_graph"] = True

    t0 = time.perf_counter()
    try:
        engine = sgl.Engine(**kwargs)
    except Exception as exc:                       # noqa: BLE001
        print(f"❌ 引擎启动失败：{type(exc).__name__}")
        print(f"   {str(exc)[:400]}")
        raise SystemExit(1)
    load_s = time.perf_counter() - t0
    print(f"\n✅ 引擎启动成功，用时 {load_s:.1f}s")

    # 打印显存账本（SGLang 会把自己的分池情况打到日志里，这里再主动取一次）
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        print(f"   启动后 CUDA 可用 {free/1024**2:.0f} MiB / 总 {total/1024**2:.0f} MiB")
    except Exception:                              # noqa: BLE001
        pass

    # ---- 单图推理 ----
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{args.question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    sampling = {"temperature": 0.0, "max_new_tokens": args.max_new_tokens}

    # 【API 差异】SGLang 的签名是
    #   generate(prompt, sampling_params, input_ids, image_data, audio_data, video_data, ...)
    # 图片必须走关键字参数 image_data=（第二位置参数是 sampling_params）。
    # 这与 vLLM 的 {"prompt": ..., "multi_modal_data": {"image": ...}} 完全不同 ——
    # 我第一版就是按 vLLM 的习惯传的，直接 TypeError。
    print("\n--- 第 1 次（冷启动）---")
    t1 = time.perf_counter()
    try:
        out = engine.generate(prompt, sampling, image_data=image)
    except Exception as exc:                       # noqa: BLE001
        print(f"❌ 推理失败：{type(exc).__name__}")
        print(f"   {str(exc)[:400]}")
        raise SystemExit(2)
    dt1 = time.perf_counter() - t1
    text = out.get("text", "") if isinstance(out, dict) else str(out)
    print(f"  ⏱ {dt1:.2f}s  输出: {text[:80]!r}")

    # ---- 第 2 次（同一图片与问题，前缀应命中）----
    print("\n--- 第 2 次（同前缀，应命中）---")
    t2 = time.perf_counter()
    out2 = engine.generate(prompt, sampling, image_data=image)
    dt2 = time.perf_counter() - t2
    text2 = out2.get("text", "") if isinstance(out2, dict) else str(out2)
    print(f"  ⏱ {dt2:.2f}s  输出: {text2[:80]!r}")

    # 看能否拿到 prefix cache 命中指标
    print("\n--- prefix cache 指标 ---")
    for attr in ("get_server_info", "server_info"):
        if hasattr(engine, attr):
            try:
                info = getattr(engine, attr)()
                if isinstance(info, dict):
                    for k, v in info.items():
                        if any(t in k.lower() for t in ("cache", "token", "mem", "radix")):
                            print(f"  {k} = {v}")
                    break
            except Exception as exc:               # noqa: BLE001
                print(f"  ⚠️ {attr}: {str(exc)[:100]}")

    print()
    print("=== 汇总 ===")
    print(f"  引擎启动   : {load_s:.1f}s")
    print(f"  首条延迟   : {dt1:.2f}s（含一次性开销）")
    print(f"  第二条延迟 : {dt2:.2f}s")
    if dt2 > 0:
        print(f"  同前缀加速 : {dt1/dt2:.2f}×")
    print()
    print("SGLANG_SMOKE_OK")

    engine.shutdown()


if __name__ == "__main__":
    main()
