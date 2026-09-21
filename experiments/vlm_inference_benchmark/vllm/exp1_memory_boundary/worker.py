"""exp1 单配置执行器：用给定参数起一次 vLLM，跑几条请求，输出一行 JSON 结果。

为什么单独一个进程跑一个配置：
  CUDA 上下文与显存池**无法在同一进程里干净复位**，而且一次 OOM 会污染进程状态，
  让后续配置的结果不可信。子进程隔离保证每个配置都是"干净开机"。

为什么显存峰值用 NVML 整卡口径：
  vLLM 0.28 的推理核心在 EngineCore 子进程里，本进程调 `torch.cuda.*` 只能看到 ≈0。
  详见 shared/common.py 里 NvmlPeakSampler 的说明。

输出约定：
  - 最后一行打印 `WORKER_RESULT {json}`，供 sweep.py 解析；
  - 引擎自身的日志（KV 池大小、non_kv 明细）走 stderr，由 sweep.py 用正则抓。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 路径锚点：向上两级到 vlm_inference_benchmark，再 import shared
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.common import (  # noqa: E402
    DEFAULT_IMAGE,
    MODEL_AWQ,
    NvmlPeakSampler,
    build_image_prompt,
)

# 这几个环境变量是实测出来的必需项，缺一个就起不来或结果不可信
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # 否则 flashinfer JIT 编译 sampling op 失败
os.environ.setdefault("HF_HUB_OFFLINE", "1")               # 模型已下好，禁止联网
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    ap.add_argument("--cpu-offload-gb", type=float, default=0.0)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--max-num-seqs", type=int, default=1)
    ap.add_argument("--max-num-batched-tokens", type=int, default=0,
                    help="0 = 交给 vLLM 从 max_model_len×max_num_seqs 推导")
    ap.add_argument("--max-pixels", type=int, default=262144)
    ap.add_argument("--num-requests", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--model", default=MODEL_AWQ)
    args = ap.parse_args()

    sampler = NvmlPeakSampler()
    base = sampler.start()

    result: dict = {
        "ok": False,
        "error": "",
        "base_mib": base,
        "peak_mib": 0,
        "delta_mib": 0,
        "load_s": 0.0,
        "gen_s": 0.0,
        "output_tokens": 0,
        "kv_gib": 0.0,
        "kv_tokens": 0,
        "non_kv_mib": 0,
        "weights_mib": 0,
        "activation_mib": 0,
        "cudagraph_mib": 0,
    }
    rc = 0

    try:
        from PIL import Image
        from vllm import LLM, SamplingParams

        kwargs: dict = dict(
            model=args.model,
            quantization="awq",
            cpu_offload_gb=args.cpu_offload_gb,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            limit_mm_per_prompt={"image": 1, "video": 0},
            mm_processor_kwargs={"min_pixels": 3136, "max_pixels": args.max_pixels},
        )
        # 只有显式给了才传，否则让 vLLM 自己推导（这也是被扫描的变量之一）
        if args.max_num_batched_tokens > 0:
            kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens

        t0 = time.perf_counter()
        llm = LLM(**kwargs)
        result["load_s"] = round(time.perf_counter() - t0, 2)

        image = Image.open(DEFAULT_IMAGE).convert("RGB")
        prompt = build_image_prompt("请用一句话描述这张图的内容。")
        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

        t1 = time.perf_counter()
        outs = llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": image}}] * args.num_requests,
            sampling, use_tqdm=False)
        result["gen_s"] = round(time.perf_counter() - t1, 3)
        result["output_tokens"] = sum(len(o.outputs[0].token_ids) for o in outs)
        result["ok"] = True

    except Exception as exc:                       # noqa: BLE001
        # 失败也要把显存读数带回去 —— "失败时的占用"本身就是边界信息
        result["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        rc = 1

    snapshot = sampler.stop()
    result["peak_mib"] = snapshot["peak_mib"]
    result["delta_mib"] = snapshot["delta_mib"]

    print("WORKER_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())