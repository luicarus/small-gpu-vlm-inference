"""执行器：prefix caching 开/关对比（同一份共享前缀 workload）。

## 设计要点
1. **指标优先看 `prefill_time`**：prefix caching 只省 prefill，
   总时长里混着 decode（完全不受影响），会稀释效应。
   vLLM 的 `RequestStateStats.prefill_time` 直接给出这段时间（`v1/metrics/stats.py:250`）。
2. **同时记录命中率**：`RequestOutput.num_cached_tokens / len(prompt_token_ids)`，
   这是解释加速比的直接证据（不是从时间反推）。
3. **max_num_seqs=1 串行**：本实验只关心「前一个请求的 KV 能否被后一个复用」，
   并发会把变量搅浑（并发时块可能被抢占/淘汰）。
4. **首个请求单独看**：缓存是冷的，命中必然为 0——它是"没有缓存时的 prefill 代价"的天然对照。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")


def build_prompt(question: str) -> str:
    """Qwen2-VL 对话模板；视频用 `<|video_pad|>` 占位（图片才是 `<|image_pad|>`）。"""
    return (
        "<|im_start|>user\n"
        "<|vision_start|><|video_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", type=Path,
                    default=Path(__file__).resolve().parent / "workload.json")
    ap.add_argument("--out", required=True, help="逐条结果 JSONL")
    ap.add_argument("--model", default=str(Path.home() / "models/Qwen2-VL-2B-Instruct-AWQ"))
    ap.add_argument("--num-questions", type=int, default=20)
    ap.add_argument("--frames", type=int, default=0, help="0 = 用 workload 里的帧数")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=4096,
                    help="视频 prompt 约 2600 token，必须大于 1024")
    ap.add_argument("--max-pixels", type=int, default=262144)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    ap.add_argument("--no-prefix-caching", action="store_true",
                    help="关掉 prefix caching（对照组）")
    args = ap.parse_args()

    from PIL import Image
    from vllm import LLM, SamplingParams

    spec = json.loads(args.workload.read_text(encoding="utf-8"))

    def resolve_image_path(raw: str) -> str:
        """workload.json 里存的是相对仓库根的路径（可移植）；这里补回绝对前缀。"""
        p = Path(raw)
        return str(p if p.is_absolute() else INFRA_ROOT / p)

    spec["image"] = resolve_image_path(spec["image"])
    n_frames = args.frames or spec["frames"]
    image = Image.open(spec["image"]).convert("RGB")
    video = [image] * n_frames          # list[PIL.Image] 即 vLLM 接受的 video 输入
    questions = spec["questions"][: args.num_questions]

    tag = "off" if args.no_prefix_caching else "on"
    print(f"prefix caching = {tag}   N = {len(questions)}   帧数 = {n_frames}")

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
        disable_log_stats=False,   # 必须开，否则拿不到 prefill_time 等逐请求指标
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    print(f"{'#':>3} {'prompt':>7} {'cached':>7} {'命中%':>7} "
          f"{'wall s':>8} {'ttft*':>8} {'out_tok':>8}")
    print("-" * 68)

    records = []
    t_run0 = time.perf_counter()
    for i, q in enumerate(questions):
        # 【为什么自己计时】vLLM 的 first_token_latency 用
        #   iteration_timestamp（引擎核心迭代时钟）− arrival_time（事件时间戳）
        # 两个时间戳来源不同，实测出现**负数**（-2.0s）——指标不可信。
        # 同理 prefill_time / e2e_latency 恒为 0（事件未填充）。
        # 所以主指标改用本进程的 perf_counter 墙钟；vLLM 指标只作参考并打星号。
        t0 = time.perf_counter()
        out = llm.generate(
            [{"prompt": build_prompt(q), "multi_modal_data": {"video": video}}],
            sampling, use_tqdm=False)[0]
        wall = time.perf_counter() - t0

        n_prompt = len(out.prompt_token_ids)
        n_cached = out.num_cached_tokens or 0
        m = out.metrics
        rec = {
            "i": i,
            "question": q,
            "prompt_tokens": n_prompt,
            "cached_tokens": n_cached,
            "hit_rate": round(n_cached / n_prompt, 4) if n_prompt else 0.0,
            "wall_s": round(wall, 4),          # ⭐ 主指标：本进程墙钟
            "ttft_vllm_s": round(getattr(m, "first_token_latency", 0.0) or 0.0, 4),
            "prefill_vllm_s": round(getattr(m, "prefill_time", 0.0) or 0.0, 4),
            "e2e_vllm_s": round(getattr(m, "e2e_latency", 0.0) or 0.0, 4),
            "output_tokens": len(out.outputs[0].token_ids),
            "error": "",
        }
        records.append(rec)
        print(f"{i:>3} {n_prompt:>7} {n_cached:>7} {rec['hit_rate'] * 100:>6.1f}% "
              f"{rec['wall_s']:>8.3f} {rec['ttft_vllm_s']:>8.3f} {rec['output_tokens']:>8}")

    wall = time.perf_counter() - t_run0

    # ---- 汇总：把首个请求（冷缓存 + 一次性预热）与后续（命中）分开统计 ----
    cold = records[0]
    warm = records[1:] or [cold]

    def mean(key, rows):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def median(key, rows):
        vals = sorted(r[key] for r in rows)
        return vals[len(vals) // 2] if vals else 0.0

    summary = {
        "prefix_caching": tag,
        "n_questions": len(questions),
        "frames": n_frames,
        "max_model_len": args.max_model_len,
        "wall_time_s": round(wall, 2),
        "cold": {"wall_s": cold["wall_s"], "prompt_tokens": cold["prompt_tokens"]},
        "warm": {"mean_wall_s": round(mean("wall_s", warm), 4),
                 "median_wall_s": round(median("wall_s", warm), 4),
                 "min_wall_s": round(min(r["wall_s"] for r in warm), 4),
                 "max_wall_s": round(max(r["wall_s"] for r in warm), 4),
                 "mean_hit_rate": round(mean("hit_rate", warm), 4),
                 "mean_cached_tokens": round(mean("cached_tokens", warm), 1),
                 "mean_prompt_tokens": round(mean("prompt_tokens", warm), 1),
                 "mean_ttft_vllm_s": round(mean("ttft_vllm_s", warm), 4)},
        "total_output_tokens": sum(r["output_tokens"] for r in records),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write(json.dumps({"_meta": summary}, ensure_ascii=False) + "\n")
        for r in records:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    print("=== 汇总（指标=本进程墙钟，含 16 个输出 token 的 decode）===")
    print(f"  冷启动（#0）  wall {cold['wall_s']:.3f}s")
    w = summary["warm"]
    print(f"  后续（命中）  wall 均值 {w['mean_wall_s']:.3f}s / 中位 {w['median_wall_s']:.3f}s "
          f"/ 范围 [{w['min_wall_s']:.3f}, {w['max_wall_s']:.3f}]")
    print(f"  平均命中率    {w['mean_hit_rate'] * 100:.1f}%"
          f"（{w['mean_cached_tokens']:.0f} / {w['mean_prompt_tokens']:.0f} token）")
    print(f"  参考：vLLM 自报 ttft 均值 {w['mean_ttft_vllm_s']:.3f}s "
          f"（⚠️ 该指标实测会出负数，仅供参考）")
    print(f"  整轮墙钟      {wall:.1f}s")
    print(f"逐条结果：{out_path}")
    print("BENCH_RUN_OK")


if __name__ == "__main__":
    main()