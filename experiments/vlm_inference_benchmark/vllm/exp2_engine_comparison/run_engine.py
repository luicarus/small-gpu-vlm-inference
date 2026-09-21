"""exp2：Qwen2-VL-2B 双栈执行器 —— 遍历 OCRBench 子集，逐条记录延迟/吞吐/输出。

设计要点：
1. **模型类自动分发**：按 config.json 的 architectures 选择 transformers 类
   （Qwen2-VL 与 Qwen2.5-VL 是**不同的类**，硬编码会让另一个直接加载失败）
2. **默认模型 2B**，且 vLLM 侧默认 `cpu_offload_gb=0`（权重 2.74 GiB 装得下）
   —— 关掉 offload 是本实验成立的前提：否则解码走 PCIe（~7 GB/s），
   测出来的是"谁被迫 offload"而不是引擎差距。
3. **HF 侧不跳过 ViT**——接受量化范围不对称。
   理由：①「HF 侧 ViT 保持 bf16」被框架缺陷堵死（混精度推理抛 AssertionError，实测非显存问题）；
   ② AWQ 不量化 ViT 是其固有特性，与 NF4 全量化的差异**正是本实验要比较的对象**。

测量口径：NVML 整卡峰值 + 贪心解码 + 同像素预算，保证与 exp3 及历史数据可比。
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path


class DeviceMemorySampler:
    """整卡显存峰值采样（NVML 口径）——与主实验逐字一致。"""

    INTERVAL_S = 0.2

    def __init__(self) -> None:
        self.base_used_mib: float | None = None
        self.peak_used_mib: float | None = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as exc:
            print(f"[sampler] NVML unavailable: {exc}")
            return
        while not self._stop_event.is_set():
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used = info.used / 1024 / 1024
            if self.base_used_mib is None:
                self.base_used_mib = used
            if self.peak_used_mib is None or used > self.peak_used_mib:
                self.peak_used_mib = used
            self._stop_event.wait(self.INTERVAL_S)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)


def load_items(path: Path, limit: int) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items[:limit] if limit else items


def check_base_memory(threshold_mib: int) -> float:
    """【基线守卫】跑之前确认整卡干净。

    2026-09-18 踩坑：一次冒烟在基线 4069 MiB（只剩 27 MiB）的污染环境下开跑，
    失败结果事后无法区分「真 bug」与「被掩盖的 OOM」，整轮作废。
    桌面占用正常范围 134~800 MiB，超过阈值直接拒绝。
    """
    import pynvml

    pynvml.nvmlInit()
    used = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).used / 1024 / 1024
    if used > threshold_mib:
        raise SystemExit(
            f"❌ 基线守卫拦截：整卡已用 {used:.0f} MiB > 阈值 {threshold_mib} MiB。\n"
            "   多半是 WDDM 显存泄漏（进程已死但驱动未回收）。\n"
            "   处理：Win+Ctrl+Shift+B 重启显卡驱动；无效则重启 Windows。"
        )
    return used


def pick_model_class(model_path: str):
    """按 config.json 的 architectures 选 transformers 类。

    Qwen2-VL 与 Qwen2.5-VL 是不同类；硬编码其中一个会让另一个加载失败。
    """
    import transformers

    arch = json.loads(
        (Path(model_path) / "config.json").read_text(encoding="utf-8")
    )["architectures"][0]
    cls = getattr(transformers, arch, None)
    if cls is None:
        raise ValueError(f"transformers 里找不到架构 {arch}")
    print(f"[model] 架构 {arch} → {cls.__name__}")
    return cls


def build_vllm(args):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # 必须关掉 flashinfer sampler：本机 JIT 编译 sampling op 会失败（实测）
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.vllm_model,
        quantization="awq",
        cpu_offload_gb=args.cpu_offload_gb,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=max(args.max_num_seqs, args.batch_size),
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
        **({"max_num_batched_tokens": args.max_num_batched_tokens}
           if args.max_num_batched_tokens > 0 else {}),
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    def generate(prompts: list[str], images: list) -> list[str]:
        inputs = [{"prompt": p, "multi_modal_data": {"image": im}}
                  for p, im in zip(prompts, images)]
        outputs = llm.generate(inputs, sampling, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    meta = {
        "engine": "vllm", "quant": "awq", "model": args.vllm_model,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cpu_offload_gb": args.cpu_offload_gb,
        "max_model_len": args.max_model_len,
        "vision_quantized": False,   # AWQ 权重保持 visual 为 bf16
    }
    return generate, meta


def build_hf(args):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig

    quant_config = None
    if args.hf_quant == "nf4":
        # 【方案 A】不传 llm_int8_skip_modules —— 即 ViT 也量化。
        # 传了会让推理抛 AssertionError（transformers 5.16.1 + bnb 0.50.2 缺陷），实测 3B/2B 均复现。
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model_cls = pick_model_class(args.hf_model)
    model = model_cls.from_pretrained(
        args.hf_model,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(
        args.hf_model, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )
    processor.tokenizer.padding_side = "left"   # 批量生成必须左填充（否则 padding 会破坏对齐）

    def generate(prompts: list[str], images: list) -> list[str]:
        texts = [
            processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "image", "image": im},
                    {"type": "text", "text": q},
                ]}],
                tokenize=False, add_generation_prompt=True,
            )
            for im, q in zip(images, prompts)
        ]
        inputs = processor(
            text=texts, images=images, return_tensors="pt",
            padding=len(texts) > 1,
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        new = out[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(new, skip_special_tokens=True)

    meta = {"engine": "hf", "quant": args.hf_quant, "model": args.hf_model,
            "device_map": "none", "double_quant": True,
            "vision_quantized": args.hf_quant == "nf4"}   # NF4 全量化 = ViT 也量化
    return generate, meta


def make_vllm_prompt(question: str) -> str:
    """vLLM 侧手写模板（HF 侧走 apply_chat_template，两者等价）。"""
    return (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def normalize(text: str) -> str:
    """答案标准化：小写、压缩空白、去常见标点（保证跨引擎一致率可比）。"""
    import re

    t = text.strip().lower()
    t = re.sub(r"[\s]+", " ", t)
    return t.strip(" .,:;!?\"'`*#")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["vllm", "hf"], required=True)
    parser.add_argument("--items", default=str(Path.home() / "datasets/OCRBench/subset100/items.jsonl"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--max-base-mib", type=float, default=1000,
                        help="基线守卫阈值：整卡占用超过它直接拒绝运行")
    # vLLM（2B 默认：offload=0、gmu 0.80）
    parser.add_argument("--vllm-model", default=str(Path.home() / "models/Qwen2-VL-2B-Instruct-AWQ"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    # HF
    parser.add_argument("--hf-model", default=str(Path.home() / "models/Qwen2-VL-2B-Instruct"))
    parser.add_argument("--hf-quant", choices=["nf4", "none"], default="nf4")
    args = parser.parse_args()

    from PIL import Image

    items = load_items(Path(args.items), args.limit)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base = check_base_memory(int(args.max_base_mib))
    print(f"引擎={args.engine}  条数={len(items)}  batch={args.batch_size}  "
          f"像素=[{args.min_pixels},{args.max_pixels}]  跑前整卡={base:.0f} MiB")

    sampler = DeviceMemorySampler()
    sampler.start()

    t_load0 = time.perf_counter()
    generate, meta = build_vllm(args) if args.engine == "vllm" else build_hf(args)
    load_time = time.perf_counter() - t_load0
    print(f"模型加载完成：{load_time:.1f} s")

    records = []
    t_run0 = time.perf_counter()
    for start in range(0, len(items), args.batch_size):
        chunk = items[start:start + args.batch_size]
        images = [Image.open(it["image"]).convert("RGB") for it in chunk]
        questions = [it["question"] for it in chunk]
        prompts = ([make_vllm_prompt(q) for q in questions]
                   if args.engine == "vllm" else questions)

        t0 = time.perf_counter()
        error = ""
        try:
            texts = generate(prompts, images)
        except Exception as exc:   # 单条失败不中断整轮
            texts = [""] * len(chunk)
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
        dt = time.perf_counter() - t0

        for it, text in zip(chunk, texts):
            records.append({
                **{k: it[k] for k in ("id", "dataset", "question_type", "question", "answers")},
                "output": text,
                "output_norm": normalize(text),
                "latency_s": round(dt / len(chunk), 3),
                "error": error,
            })
        done = start + len(chunk)
        if done % 20 == 0 or done == len(items):
            print(f"  {done}/{len(items)}  最近一批 {dt:.2f}s")

    total_time = time.perf_counter() - t_run0
    sampler.stop()

    run_meta = {
        "engine": args.engine, "batch_size": args.batch_size,
        "items": len(items), "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
        "load_time_s": round(load_time, 2),
        "run_time_s": round(total_time, 2),
        "wall_time_s": round(load_time + total_time, 2),
        "base_mib": round(sampler.base_used_mib) if sampler.base_used_mib else None,
        "peak_mib": round(sampler.peak_used_mib) if sampler.peak_used_mib else None,
        **meta,
    }

    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write(json.dumps({"_meta": run_meta}, ensure_ascii=False) + "\n")
        for r in records:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    errors = sum(1 for r in records if r["error"])
    print()
    print("=== 运行汇总 ===")
    for k, v in run_meta.items():
        print(f"  {k}: {v}")
    print(f"  失败条数: {errors}/{len(records)}")
    print(f"逐条结果：{out_path}")
    print("S1_ENGINE_RUN_OK" if errors == 0 else "S1_ENGINE_RUN_PARTIAL")


if __name__ == "__main__":
    main()