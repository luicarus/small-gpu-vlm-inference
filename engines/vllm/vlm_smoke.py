"""Minimal single-image smoke test for a Qwen2-VL model served by vLLM."""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path


DEFAULT_MODEL = str(Path.home() / "models/Qwen2-VL-2B-Instruct-AWQ")
# 路径层级：engines/vllm/x.py → [0]=vllm [1]=engines [2]=small-gpu-vlm-inference [3]=InfraStudy
# 图片归档在 docs/assets/
DEFAULT_IMAGE = (
    Path(__file__).resolve().parents[2] / "assets" / "test_image.png"
)
DEFAULT_QUESTION = "请解释这张图中 logical block、block table 和 physical block 的关系。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("VLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1,
        help="一次送入的请求数；>1 用于验证 max_num_seqs 的并发能力",
    )
    # 【实测要点】chunked prefill 下 vLLM 会自动推导 max_num_batched_tokens
    # （≈ max_model_len × max_num_seqs），而它决定 profiling 那次 forward 的
    # 激活值峰值 —— 实测 len=2048 抬升 0.62 GiB、seqs=4 抬升 0.90 GiB，直接把 KV 池吃成负数。
    # 显式限小它，就能在保留长上下文的同时把激活值压回去。
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=0,
        help="显式限制单批 token 数；0 = 交给 vLLM 自动推导",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=50176)
    parser.add_argument("--quantization", default="awq")
    parser.add_argument(
        "--cpu-offload-gb",
        type=float,
        default=float(os.environ.get("VLM_CPU_OFFLOAD_GB", "1.0")),
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--online", action="store_true", help="Allow Hugging Face downloads")
    return parser.parse_args()


class DeviceMemorySampler:
    """【实测要点】整卡显存峰值采样器（NVML 口径，读数等同 nvidia-smi）。

    为什么不用 torch.cuda.max_memory_allocated()？两个原因：
    1. vLLM 0.28 是 V1 架构，推理核心在 EngineCore *子进程* 里跑，
       父进程的 torch 显存计数器看不到子进程的分配，读出来≈0，是经典坑；
    2. 在父进程里调 torch.cuda.* 反而会额外创建一个 CUDA context，
       白白吃掉几百 MiB 显存，在 4GB 卡上会污染测量、甚至诱发假 OOM。

    所以走 NVML（驱动层接口）：设备级读数、不建 CUDA context、不占显存。
    """

    INTERVAL_S = 0.2  # 权重加载的峰值可能只持续几秒，采样间隔太大会漏掉真实峰值

    def __init__(self) -> None:
        self.base_used_mib: float | None = None  # 启动前基线（含 Windows 桌面占用）
        self.peak_used_mib: float | None = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            import pynvml  # vllm 的依赖自带，无需额外 pip install

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 单卡机器，固定 0 号
        except Exception as exc:  # NVML 不可用时静默降级，不阻塞冒烟测试
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


def main() -> None:
    args = parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"image not found: {args.image}")

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from PIL import Image
    from vllm import LLM, SamplingParams

    # 【要点】采样必须在 LLM 初始化之前启动：权重加载 + 显存 profiling 阶段就是
    # 峰值时刻，只在 generate 期间采样会漏掉启动峰值。
    sampler = DeviceMemorySampler()
    sampler.start()

    image = Image.open(args.image).convert("RGB")

    llm_kwargs = dict(
        model=args.model,
        quantization=args.quantization,
        cpu_offload_gb=args.cpu_offload_gb,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        },
        enforce_eager=args.enforce_eager,
    )
    if args.max_num_batched_tokens > 0:
        # 只在显式指定时才传，避免改变默认行为（否则历史数据不可比）
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    llm = LLM(**llm_kwargs)

    # 【要点】并发验证：num_requests > 1 时构造多条**内容不同**的提问。
    # 为什么必须不同？相同 prompt 会被 prefix caching 命中、共享 KV block，
    # 那样测到的是前缀复用而不是并发调度。图片仍复用同一张。
    FOCUS = ["logical block", "block table", "physical block", "引用计数"]
    prompts = []
    for i in range(args.num_requests):
        if args.num_requests == 1:
            question = args.question
        else:
            question = f"{args.question} 请重点解释「{FOCUS[i % len(FOCUS)]}」。"
        prompts.append(
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    inputs = [{"prompt": p, "multi_modal_data": {"image": image}} for p in prompts]

    gen_t0 = time.perf_counter()
    outputs = llm.generate(
        inputs,
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        use_tqdm=False,
    )
    gen_time = time.perf_counter() - gen_t0

    result = outputs[0].outputs[0].text
    sampler.stop()
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    # 【要点】整卡口径峰值（含 ~300MiB 桌面/驱动基线）；
    # DELTA = 峰值 - 基线 ≈ 本次推理真正净增的显存占用，跨配置比较看它更公平。
    if sampler.peak_used_mib is not None:
        print(f"GPU_MEM_BASE_MIB: {sampler.base_used_mib:.0f}")
        print(f"GPU_MEM_PEAK_MIB: {sampler.peak_used_mib:.0f}")
        print(f"GPU_MEM_PEAK_DELTA_MIB: {sampler.peak_used_mib - sampler.base_used_mib:.0f}")
    else:
        print("GPU_MEM_PEAK_MIB: N/A (pynvml 不可用)")

    print("MODEL:", args.model)
    print("IMAGE:", args.image)
    print("GPU_MEMORY_UTILIZATION:", args.gpu_memory_utilization)
    print("MAX_MODEL_LEN:", args.max_model_len)
    print("MAX_NUM_SEQS:", args.max_num_seqs)
    print("CPU_OFFLOAD_GB:", args.cpu_offload_gb)
    print("NUM_REQUESTS:", args.num_requests)
    # 【要点】吞吐三件套：扫描脚本靠它们做 offload 的「带宽账」校准
    print("GEN_TIME_S:", round(gen_time, 2))
    print("OUTPUT_TOKENS:", total_tokens)
    print("TOKENS_PER_S:", round(total_tokens / gen_time, 2) if gen_time > 0 else 0.0)
    print("QUESTION:", args.question)
    print("ANSWER:", result)
    print("VLM_SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
