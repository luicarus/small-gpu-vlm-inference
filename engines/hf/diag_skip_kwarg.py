"""诊断 v2：找对「让 ViT 保持 bf16」的正确参数名。

v1 用了 `llm_int8_skip_modules=["visual"]`：
  * ViT 仍是 4bit（uint8 参数量 1720.6M 完全没变）
  * 峰值却从 2349 → 4058 MiB
→ 说明参数名在 transformers 5.16 里可能已改名（`llm_int8_skip_modules` 是旧名）。

本脚本不猜：先打印 BitsAndBytesConfig 的**真实字段**，再逐个试候选参数名。
"""

from __future__ import annotations

import os
from collections import Counter

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

MODEL = str(Path.home()) + "/models/Qwen2-VL-2B-Instruct"


def census(model) -> tuple[int, float, float]:
    """返回 (Linear4bit 模块数, 4bit(uint8) 参数量 M, bf16 参数量 M)。"""
    n_lin4 = sum(1 for m in model.modules() if type(m).__name__ == "Linear4bit")
    n_other_lin = Counter(type(m).__name__ for m in model.modules()
                          if "Linear" in type(m).__name__ and type(m).__name__ != "Linear4bit")
    m_u8 = sum(p.numel() for p in model.parameters() if p.dtype == torch.uint8) / 1e6
    m_bf = sum(p.numel() for p in model.parameters() if p.dtype == torch.bfloat16) / 1e6
    return n_lin4, m_u8, m_bf, dict(n_other_lin)


def main() -> None:
    print("=== 1. BitsAndBytesConfig 的真实字段（看有没有 skip / not_convert 类字段）===")
    cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    fields = list(cfg.to_dict().keys())
    for f in fields:
        if any(s in f for s in ("skip", "convert", "module", "int8", "4bit", "quant_type")):
            print(f"    {f} = {getattr(cfg, f, None)}")
    print(f"    （共 {len(fields)} 个字段）")

    print()
    print("=== 2. 模型的顶层子模块名（确认 ViT 到底叫什么）===")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    print(f"    {[n for n, _ in model.named_children()]}")
    vit = getattr(model, "visual", None)
    if vit is not None:
        print(f"    visual 的子模块: {[n for n, _ in vit.named_children()][:8]} ...")
        print(f"    visual 参数量: {sum(p.numel() for p in vit.parameters()) / 1e6:.1f} M")
    del model
    torch.cuda.empty_cache()

    print()
    print("=== 3. 逐个尝试候选参数名 ===")
    print(f"{'候选':<28} {'Linear4bit':>10} {'uint8 M':>9} {'bf16 M':>8} {'峰值 MiB':>9}  其他 Linear")
    print("-" * 100)

    candidates: list[tuple[str, dict]] = [
        ("(基线) 不 skip", {}),
        ("llm_int8_skip_modules", {"llm_int8_skip_modules": ["visual"]}),
        ("modules_to_not_convert", {"modules_to_not_convert": ["visual"]}),
        ("modules_to_not_convert=model.visual", {"modules_to_not_convert": ["model.visual"]}),
    ]

    for name, extra in candidates:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        try:
            cfg = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                **extra)
            m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL, torch_dtype=torch.bfloat16, quantization_config=cfg,
                low_cpu_mem_usage=True, attn_implementation="sdpa")
            m = m.to("cuda")
            torch.cuda.synchronize()
            n4, u8, bf, other = census(m)
            peak = torch.cuda.max_memory_allocated() / 1024**2
            print(f"{name:<28} {n4:>10} {u8:>9.1f} {bf:>8.1f} {peak:>9.0f}  {other}")
            del m
        except Exception as exc:
            print(f"{name:<28} 失败: {type(exc).__name__}: {str(exc)[:60]}")
        torch.cuda.empty_cache()

    print()
    print("=== 判读 ===")
    print("  ViT 若真的保持 bf16：Linear4bit 数量应减少约 128 个（ViT 的 32 层 × 4 个线性层），")
    print("  且 bf16 参数量应从 313M 涨到约 760M（+0.44B ViT 权重）。")
    print("DIAG2_DONE")


if __name__ == "__main__":
    main()