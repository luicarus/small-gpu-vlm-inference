"""冒烟：验证「伪视频共享前缀」这条路能跑通，并测出三个关键未知量。

要回答的三个问题（都是正式实验的设计前提）：
  1. **视频路径能跑通吗**？—— vLLM 的 Qwen2-VL 接受 `list[PIL.Image]` 作为 video 输入，
     prompt 里用 `<|video_pad|>` 占位。这条路径本次是第一次用，必须先验证。
  2. **共享前缀到底多少 token**？—— 决定理论上限加速比 `(P+Q)/Q`。
  3. **prefix caching 真的命中吗、命中了多少**？—— 用 vLLM 直接暴露的
     `RequestOutput.num_cached_tokens`（`vllm/outputs.py:120`）精确测量，不靠时间反推。

跑法：同一份伪视频 + 3 个不同问题，串行送入，打印每个请求的
prompt token 数 / 命中 token 数 / 命中率。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 路径锚点走 shared：workload.json 里的图片是相对仓库根的路径，解析时需要根目录
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.common import INFRA_ROOT  # noqa: E402

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

WORKLOAD = Path(__file__).resolve().parent / "workload.json"


def resolve_image_path(raw: str) -> str:
    """workload.json 里存的是相对仓库根的路径（可移植）；这里补回绝对前缀。"""
    p = Path(raw)
    return str(p if p.is_absolute() else INFRA_ROOT / p)


def build_prompt(question: str) -> str:
    """Qwen2-VL 的对话模板 + 视频占位符。

    注意与图片的差别：图片用 `<|image_pad|>`，视频用 `<|video_pad|>`
    （源码 `model_executor/models/qwen2_vl.py:1279`）。
    """
    return (
        "<|im_start|>user\n"
        "<|vision_start|><|video_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(Path.home() / "models/Qwen2-VL-2B-Instruct-AWQ"))
    ap.add_argument("--workload", type=Path, default=WORKLOAD)
    ap.add_argument("--num-questions", type=int, default=3)
    ap.add_argument("--frames", type=int, default=0, help="覆盖 workload 里的帧数")
    ap.add_argument("--max-model-len", type=int, default=4096,
                    help="视频 token 多，需要比 1024 更大的上下文")
    ap.add_argument("--max-pixels", type=int, default=262144)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    ap.add_argument("--no-prefix-caching", action="store_true",
                    help="关掉 prefix caching（做对照用的基线）")
    ap.add_argument("--max-tokens", type=int, default=16)
    args = ap.parse_args()

    from PIL import Image
    from vllm import LLM, SamplingParams

    spec = json.loads(args.workload.read_text(encoding="utf-8"))
    spec["image"] = resolve_image_path(spec["image"])
    n_frames = args.frames or spec["frames"]
    image = Image.open(spec["image"]).convert("RGB")
    # 伪视频：同一张图重复 K 帧（list[PIL.Image] 是 vLLM 接受的 video 输入形式）
    video = [image] * n_frames
    questions = spec["questions"][: args.num_questions]

    print(f"模型    : {args.model}")
    print(f"伪视频  : {n_frames} 帧（同一张 {image.size} 的图重复）")
    print(f"问题数  : {len(questions)}")
    print(f"prefix caching: {'关闭（对照）' if args.no_prefix_caching else '开启（默认）'}")
    print()

    llm = LLM(
        model=args.model,
        quantization="awq",
        cpu_offload_gb=0.0,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        limit_mm_per_prompt={"image": 0, "video": 1},
        mm_processor_kwargs={"min_pixels": 3136, "max_pixels": args.max_pixels},
        enable_prefix_caching=not args.no_prefix_caching,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    print()
    print(f"{'#':>3} {'prompt_tok':>11} {'cached_tok':>11} {'命中率':>8} {'耗时 s':>8}  回答前 30 字")
    print("-" * 84)

    rows = []
    for i, q in enumerate(questions):
        prompt = build_prompt(q)
        inputs = {"prompt": prompt, "multi_modal_data": {"video": video}}
        t0 = time.perf_counter()
        out = llm.generate([inputs], sampling, use_tqdm=False)[0]
        dt = time.perf_counter() - t0

        n_prompt = len(out.prompt_token_ids)
        n_cached = out.num_cached_tokens or 0
        hit = n_cached / n_prompt * 100 if n_prompt else 0
        text = out.outputs[0].text.replace("\n", " ")[:30]
        print(f"{i:>3} {n_prompt:>11} {n_cached:>11} {hit:>7.1f}% {dt:>8.2f}  {text}")
        rows.append({"i": i, "prompt_tokens": n_prompt, "cached_tokens": n_cached,
                     "time_s": round(dt, 3)})

    print()
    if len(rows) > 1:
        first = rows[0]
        print("=== 判读 ===")
        print(f"  首个请求 prompt 共 {first['prompt_tokens']} token、命中 {first['cached_tokens']}"
              f"（首个请求理应命中 0——缓存是空的）")
        later = rows[1:]
        avg_hit = sum(r["cached_tokens"] for r in later) / len(later)
        avg_prompt = sum(r["prompt_tokens"] for r in later) / len(later)
        print(f"  后续请求平均 prompt {avg_prompt:.0f} token、平均命中 {avg_hit:.0f} token"
              f"（命中率 {avg_hit / avg_prompt * 100:.1f}%）")
        print()
        if avg_hit > 0:
            print("  ✅ 共享前缀命中成功 → 可以跑正式实验")
            print(f"     共享前缀 P ≈ {avg_hit:.0f} token（对齐到 16 的整数倍）")
        else:
            print("  ❌ 没有命中 —— 需要排查：视频 token 是否一致 / 块是否足够大 / 是否被淘汰")
    print("SMOKE_VIDEO_DONE")


if __name__ == "__main__":
    main()