"""诊断 v3：用**正确**的模块路径让 ViT 保持 bf16，并验证推理能跑通。

v2 的两个发现：
  1. 模型顶层子模块是 ['model', 'lm_head'] → ViT 真实路径是 `model.visual`
     （之前一直写 "visual"，匹配不到任何模块）
  2. 加 skip 参数后 uint8 / bf16 参数量毫无变化 → ViT 从未被跳过

本脚本用正确路径逐个试，并且**实际跑一次 generate**——因为之前
`--keep-vision-bf16` 是在 generate 阶段报 FP4 state 未初始化，只看加载是不够的。
"""

from __future__ import annotations

import os
from collections import Counter

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

MODEL = str(Path.home()) + "/models/Qwen2-VL-2B-Instruct"
IMAGE = ("/mnt/d/Work Places/Python Work Place/Job/InfraStudy/"
         "docs/assets/test_image.png")
QUESTION = "请解释这张图中 logical block、block table 和 physical block 的关系。"


def main() -> None:
    torch.cuda.init()
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    for label, skip in [
        ("基线：全量化", None),
        ("skip=['model.visual']", ["model.visual"]),
        ("skip=['visual','model.visual']", ["visual", "model.visual"]),
    ]:
        print("=" * 80)
        print(f"### {label}")
        print("=" * 80)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        if skip:
            cfg.llm_int8_skip_modules = skip

        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL, torch_dtype=torch.bfloat16, quantization_config=cfg,
                low_cpu_mem_usage=True, attn_implementation="sdpa").to("cuda")
        except Exception as exc:
            print(f"  加载失败: {type(exc).__name__}: {str(exc)[:150]}\n")
            continue

        # --- 量化范围核查 ---
        n4 = sum(1 for m in model.modules() if type(m).__name__ == "Linear4bit")
        m_u8 = sum(p.numel() for p in model.parameters() if p.dtype == torch.uint8) / 1e6
        m_bf = sum(p.numel() for p in model.parameters() if p.dtype == torch.bfloat16) / 1e6
        vit = model.model.visual
        vit_types = Counter(type(m).__name__ for m in vit.modules()
                            if "Linear" in type(m).__name__)
        print(f"  Linear4bit 模块数 : {n4}")
        print(f"  uint8(4bit) 参数  : {m_u8:.1f} M")
        print(f"  bf16 参数         : {m_bf:.1f} M")
        print(f"  ViT 线性层类型    : {dict(vit_types)}")

        # --- 真正跑一次推理（加载成功≠能推理，之前就败在这一步）---
        try:
            processor = AutoProcessor.from_pretrained(
                MODEL, min_pixels=3136, max_pixels=262144)
            img = Image.open(IMAGE).convert("RGB")
            text = processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": QUESTION}]}],
                tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            ans = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True)[0]
            print(f"  ✅ 推理成功: {ans[:70]!r}")
        except Exception as exc:
            print(f"  ❌ 推理失败: {type(exc).__name__}: {str(exc)[:150]}")

        print(f"  torch 峰值显存: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB")
        print()
        del model
        torch.cuda.empty_cache()

    print("DIAG3_DONE")


if __name__ == "__main__":
    main()