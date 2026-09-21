"""冒烟：HuggingFace transformers + bitsandbytes NF4 跑 Qwen2.5-VL（可行性验证）。

为什么单独写、不复用 engines/vllm/vlm_smoke.py：
- vLLM 路径是 `LLM(...) + SamplingParams`，HF 路径是 `model.generate() + processor`，
  两套 API 完全不同，硬合并只会写出难读的条件分支；
- 但**测量口径必须一致**：同样用 NVML 整卡峰值采样、同样打印 `GPU_MEM_*` 与
  `TOKENS_PER_S` 等机器可解析的行。核心是两个引擎对比，
  尺子不一致的话所有结论都是假的。

与 vLLM 路径的关键差异（也决定了预期结果）：
1. **权重在同一个进程里**：没有 EngineCore 子进程，所以本进程 torch 计数器是准的，
   但为了跨引擎可比，仍统一用 NVML 整卡口径。
2. **没有 PagedAttention / continuous batching**：KV cache 按 max_new_tokens 全量预分配，
   峰值显存随输出长度线性增长——这是 HF 在 4GB 卡上最大的风险点。
3. **没有 cpu_offload_gb**：HF 侧的等价物是 `device_map="auto"` + accelerate
   （把放不下的层甩到 CPU），语义与 vLLM 的 UVA offload 不同，报告里必须写清。
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

# 路径层级：engines/hf/x.py → [0]=hf [1]=engines [2]=small-gpu-vlm-inference [3]=InfraStudy
DEFAULT_MODEL = str(Path.home()) + "/models/Qwen2-VL-2B-Instruct"
DEFAULT_IMAGE = (
    Path(__file__).resolve().parents[2] / "assets" / "test_image.png"
)
DEFAULT_QUESTION = "请解释这张图中 logical block、block table 和 physical block 的关系。"


class DeviceMemorySampler:
    """整卡显存峰值采样器（NVML 口径）——与 vLLM 路径**逐字一致**，保证跨引擎可比。

    为什么不用 torch.cuda.max_memory_allocated()：
    HF 路径下它其实可用（权重在同进程），但 vLLM 路径下它是 0（V1 引擎在子进程）。
    两个引擎要用同一把尺子，所以统一走 NVML（驱动层，不建额外 CUDA context）。
    """

    INTERVAL_S = 0.2  # 采样间隔：权重加载与首次 forward 的峰值可能只持续几秒

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
            used_mib = info.used / 1024 / 1024
            if self.base_used_mib is None:
                self.base_used_mib = used_mib
            if self.peak_used_mib is None or used_mib > self.peak_used_mib:
                self.peak_used_mib = used_mib
            self._stop_event.wait(self.INTERVAL_S)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("HF_MODEL", DEFAULT_MODEL))
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--quant",
        choices=["nf4", "none"],
        default="nf4",
        help="nf4 = bitsandbytes 4bit（主角）；none = bfloat16（用于确认不是量化本身导致 OOM）",
    )
    parser.add_argument(
        "--keep-vision-bf16",
        action="store_true",
        help="vision tower 不量化（对齐 AWQ 版的 modules_to_not_convert=['visual']）；"
        "默认连 vision 一起量化，为了先确认「最省显存能跑通」",
    )
    parser.add_argument(
        "--double-quant",
        action="store_true",
        default=True,
        help="NF4 双重量化（再省约 0.4 bit/参数），默认开",
    )
    parser.add_argument(
        "--device-map",
        default="none",
        choices=["none", "auto"],
        help="none = 全部塞 cuda:0（放不下就 OOM，正是可行性要问的）；"
        "auto = 借助 accelerate 把放不下的层甩到 CPU（HF 侧的 offload 等价物）",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=50176)
    parser.add_argument("--attn-implementation", default="sdpa",
                        choices=["sdpa", "eager", "flash_attention_2"])
    return parser.parse_args()


def build_quant_config(args: argparse.Namespace):
    """构造 bitsandbytes 量化配置。

    quant_type="nf4" 是主角：正态分布权重的最优 4bit 信息论量化
    （QLoRA 论文结论），与 AWQ 的「激活感知、保护显著通道」是两条不同路线——
    报告里要讲清这个区别，而不是只比数字。
    """
    if args.quant == "none":
        return None

    import torch
    from transformers import BitsAndBytesConfig

    kwargs = dict(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # 计算用 bf16：量化权重反量化后参与 matmul 的精度，
        # 设成 fp16 在 4GB 卡上省不了显存反而掉精度
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=args.double_quant,
    )
    if args.keep_vision_bf16:
        # 对齐 AWQ 版模型的行为（其 config.json 里 modules_to_not_convert=["visual"]）。
        #
        # 【为什么传一串候选路径而不是单个 "visual"】
        # transformers 的 skip 匹配是「模块全名精确匹配（或以 key + '.' 为前缀）」，
        # 而 ViT 的**全名因模型而异**：
        #   Qwen2.5-VL → 顶层只有 ['model','lm_head']，ViT 全名是 `model.visual`
        #   Qwen2-VL   → ViT 是顶层直接子模块，全名就是 `visual`
        # 2026-09-16 实测：对 Qwen2.5-VL 传 ["visual"] **完全匹配不到**（参数量一字未变），
        # 传 ["model.visual"] 才生效。所以这里把候选名都带上——不匹配的名字是无害的。
        kwargs["llm_int8_skip_modules"] = [
            "visual", "model.visual", "vision_model", "model.vision_model",
        ]
    return BitsAndBytesConfig(**kwargs)


def verify_vision_not_quantized(model) -> bool:
    """加载后核实 ViT 是否真的没被量化。

    为什么要这一步：skip 参数写错时**不会报错、也不会警告**，只会静默失效
    （2026-09-16 实测：参数量一字未变，峰值却涨了 1.7 GiB）。
    不主动核实就会拿着"以为对齐了"的错误配置去跑完整实验。
    """
    vit = None
    for name, module in model.named_modules():
        if name.endswith("visual") and len(list(module.children())) > 0:
            vit = module
            break
    if vit is None:
        print("[verify] ⚠️ 未找到视觉塔，无法核实量化范围")
        return False

    from collections import Counter

    types = Counter(type(m).__name__ for m in vit.modules() if "Linear" in type(m).__name__)
    n_quant = sum(v for k, v in types.items() if "4bit" in k or "FP4" in k)
    print(f"[verify] ViT 线性层类型: {dict(types)}")
    if n_quant:
        print(f"[verify] ❌ ViT 仍有 {n_quant} 个量化层 —— skip 未生效，实验不可用于对齐比较")
        return False
    print("[verify] ✅ ViT 保持未量化（量化范围已对齐 AWQ）")
    return True


def pick_model_class(model_path: str):
    """按 config.json 的 architectures 选择模型类。

    Qwen2.5-VL 与 Qwen2-VL 是**不同的类**（Qwen2_5_VL... vs Qwen2VL...），
    硬编码其中一个会让另一个直接加载失败。
    """
    import json
    from pathlib import Path

    import transformers

    arch = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))["architectures"][0]
    cls = getattr(transformers, arch, None)
    if cls is None:
        raise ValueError(f"transformers 里找不到架构 {arch}")
    print(f"[model] 架构 {arch} → 使用 {cls.__name__}")
    return cls


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"image not found: {args.image}")

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 模型已本地化，禁止联网拖慢启动

    import torch
    from PIL import Image
    from transformers import AutoProcessor

    sampler = DeviceMemorySampler()
    sampler.start()  # 必须在加载权重之前启动，否则漏掉加载期峰值

    image = Image.open(args.image).convert("RGB")
    quant_config = build_quant_config(args)
    model_cls = pick_model_class(args.model)
    status = "OK"
    error_text = ""
    vision_aligned = None

    # ---- 1) 加载模型（可行性的第一道关：权重+ViT 能否放进 4GB）----
    t_load0 = time.perf_counter()
    try:
        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            quantization_config=quant_config,
            attn_implementation=args.attn_implementation,
            low_cpu_mem_usage=True,
        )
        if args.device_map == "auto":
            load_kwargs["device_map"] = "auto"
        model = model_cls.from_pretrained(args.model, **load_kwargs)
        if args.device_map == "none":
            model = model.to("cuda")
        processor = AutoProcessor.from_pretrained(
            args.model,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        # 加载后立刻核实 ViT 是否真的没被量化（skip 失效是静默的，必须主动查）
        if args.keep_vision_bf16:
            vision_aligned = verify_vision_not_quantized(model)
    except torch.cuda.OutOfMemoryError as exc:
        status, error_text = "OOM_LOAD", str(exc)[:300]
        model = processor = None
    except Exception as exc:  # 其他加载失败（缺包、模型格式等）
        status, error_text = "ERROR_LOAD", f"{type(exc).__name__}: {str(exc)[:300]}"
        model = processor = None
    load_time = time.perf_counter() - t_load0

    gen_time = 0.0
    total_tokens = 0
    answer = ""

    # ---- 2) 推理（第二道关：激活值 + KV 峰值）----
    if model is not None:
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": args.question},
                ],
            }]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=[text], images=[image], return_tensors="pt")
            model_device = next(model.parameters()).device
            inputs = inputs.to(model_device)

            t_gen0 = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,  # 贪心解码，与 vLLM 侧 temperature=0 对齐
                )
            gen_time = time.perf_counter() - t_gen0

            # 只统计新生成 token（HF 返回的是 prompt+completion 拼接）
            new_tokens = generated[:, inputs["input_ids"].shape[1]:]
            total_tokens = int(new_tokens.shape[1])
            answer = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        except torch.cuda.OutOfMemoryError as exc:
            status, error_text = "OOM_GENERATE", str(exc)[:300]
        except Exception as exc:
            status, error_text = "ERROR_GENERATE", f"{type(exc).__name__}: {str(exc)[:300]}"

    sampler.stop()

    # ---- 3) 报表：前 3 行与 vLLM 路径同名同格式，便于对比脚本统一解析 ----
    if sampler.peak_used_mib is not None:
        print(f"GPU_MEM_BASE_MIB: {sampler.base_used_mib:.0f}")
        print(f"GPU_MEM_PEAK_MIB: {sampler.peak_used_mib:.0f}")
        print(f"GPU_MEM_PEAK_DELTA_MIB: {sampler.peak_used_mib - sampler.base_used_mib:.0f}")
    else:
        print("GPU_MEM_PEAK_MIB: N/A (pynvml 不可用)")

    print("ENGINE: hf-transformers")
    print("MODEL:", args.model)
    print("QUANT:", args.quant)
    print("KEEP_VISION_BF16:", args.keep_vision_bf16)
    print("DEVICE_MAP:", args.device_map)
    print("GEN_TIME_S:", round(gen_time, 2))
    print("OUTPUT_TOKENS:", total_tokens)
    print("TOKENS_PER_S:", round(total_tokens / gen_time, 2) if gen_time > 0 else 0.0)
    print("LOAD_TIME_S:", round(load_time, 2))
    print("QUESTION:", args.question)
    print("ANSWER:", answer)
    if status != "OK":
        print("ERROR:", error_text)
        print(f"HF_SMOKE_TEST_FAIL ({status})")
    else:
        # 与 vLLM 侧 VLM_SMOKE_TEST_OK 区分的成功标记，扫描器按引擎分别匹配
        print("HF_SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
