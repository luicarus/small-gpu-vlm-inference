"""诊断：HF + bitsandbytes NF4 能否让 ViT 保持 bf16（--keep-vision-bf16 为何失败）。

现象：加 `llm_int8_skip_modules=["visual"]` 后报
    FP4 quantization state not initialized. Please call .cuda() or .to(device) ...
    AssertionError (generate 阶段)
且峰值显存 2669 MiB 与"ViT 也量化"时（2673）几乎相同 —— 说明 ViT 并未真正保留 bf16。

本脚本不猜，直接查：加载后统计各模块类型的数量、检查 bnb 量化状态是否初始化、
并找出到底哪些子模块处于未初始化状态。
"""

from __future__ import annotations

import os
import traceback
from collections import Counter

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

MODEL = str(Path.home()) + "/models/Qwen2-VL-2B-Instruct"


def module_census(model, tag: str) -> None:
    """统计模块类型分布：看清 ViT 与 LLM 各自被换成了什么。"""
    types = Counter(type(m).__name__ for m in model.modules())
    interesting = {k: v for k, v in types.items()
                   if any(s in k for s in ("Linear", "4bit", "FP4", "NF4", "Quant"))}
    print(f"--- [{tag}] 量化相关模块类型 ---")
    for k, v in sorted(interesting.items(), key=lambda x: -x[1]):
        print(f"    {v:>5}  {k}")


def find_uninitialized(model) -> list[str]:
    """找出 Params4bit 里 quant_state 尚未初始化的参数（就是报错的根源）。"""
    bad = []
    for name, p in model.named_parameters():
        if type(p).__name__ in ("Params4bit", "ParamsNF4", "ParamsFP4"):
            if getattr(p, "quant_state", None) is None:
                bad.append(name)
    return bad


def dev_summary(model, tag: str) -> None:
    """按设备统计参数量：能看出 ViT 的权重到底在哪、是不是 bf16。"""
    counter: Counter = Counter()
    for _, p in model.named_parameters():
        counter[(str(p.device), str(p.dtype))] += p.numel()
    print(f"--- [{tag}] 参数分布（设备, dtype）---")
    for (dev, dt), n in sorted(counter.items(), key=lambda x: -x[1]):
        print(f"    {n / 1e6:8.1f} M   {dev:<10} {dt}")


def try_load(tag: str, kwargs: dict) -> None:
    print("=" * 78)
    print(f"### 尝试：{tag}")
    print(f"    kwargs: { {k: v for k, v in kwargs.items() if k != 'quantization_config'} }")
    print("=" * 78)
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL, low_cpu_mem_usage=True, **kwargs)
    except Exception as exc:
        print(f"  加载失败: {type(exc).__name__}: {str(exc)[:200]}")
        return

    module_census(model, f"{tag} / 加载后(CPU)")
    bad_before = find_uninitialized(model)
    print(f"  未初始化的 4bit 参数数量（.to 之前）: {len(bad_before)}")
    if bad_before[:3]:
        print(f"    例: {bad_before[:3]}")

    # 关键一步：官方提示要 .to(device) 才会初始化量化状态
    if kwargs.get("device_map") is None:
        model = model.to("cuda")
    torch.cuda.synchronize()

    bad_after = find_uninitialized(model)
    print(f"  未初始化的 4bit 参数数量（.to 之后）: {len(bad_after)}")
    if bad_after[:5]:
        print(f"    例: {bad_after[:5]}")

    dev_summary(model, f"{tag} / .to 之后")
    print(f"  峰值显存: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB")

    # ViT 是否真的是 bf16？直接查一个 ViT 线性层的类型
    vit = getattr(model, "visual", None)
    if vit is not None:
        lin_types = Counter(type(m).__name__ for m in vit.modules()
                            if "Linear" in type(m).__name__)
        print(f"  ViT 内部线性层类型: {dict(lin_types)}")

    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main() -> None:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    def nf4(skip=None):
        kw: dict = dict(
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            ),
            attn_implementation="sdpa",
        )
        if skip:
            kw["quantization_config"].llm_int8_skip_modules = skip
        return kw

    # 1) 基线：ViT 也量化（当前主实验用的配置）
    try_load("A: 默认（ViT 也量化）", nf4())

    # 2) 试 skip "visual"
    try_load("B: skip=['visual']", nf4(["visual"]))

    # 3) 试 skip 更多候选名（HF 里 ViT 属性可能叫 visual / vision_model / model.visual）
    try_load("C: skip=['visual','vision_model']", nf4(["visual", "vision_model"]))

    # 4) 试 device_map=auto（官方报错提示的另一条路）
    kw = nf4(["visual"])
    kw["device_map"] = "auto"
    try_load("D: skip=['visual'] + device_map=auto", kw)

    print()
    print("DIAG_DONE")


if __name__ == "__main__":
    main()