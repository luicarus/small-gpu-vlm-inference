"""exp3 负载移植到 SGLang：共享前缀（伪视频）下的 prefix caching 收益。

## 为什么要移植
vLLM 侧的 exp3 已经测出「共享前缀越长，加速比越大」（2.43× → 12.76×）。
SGLang 用**不同的数据结构**做前缀复用（radix 树 vs 块哈希链），
在同一份负载上跑一遍，才能把差异归因到机制而不是负载。

## 与 vLLM 版（`../vllm/exp3_prefix_caching/run_cache.py`）的口径对齐
| 项 | 做法 |
|---|---|
| 负载 | **同一份 `workload.json`**（同一张图重复 K 帧 + N 个问题） |
| prompt 模板 | 同一个 Qwen2-VL 模板（`<|vision_start|><|video_pad|><|vision_end|>`） |
| 延迟 | **本进程墙钟**，并**剔除第 0 条**（冷启动 + 一次性 JIT），报中位数 |
| 命中指标 | vLLM 用 `RequestOutput.num_cached_tokens`；**SGLang 用 `meta_info["cached_tokens"]`** |
| 解码 | `temperature=0`（贪心）、`max_new_tokens` 固定 |
| 显存 | 记录 KV 池大小（SGLang 会打印），**两侧不相等，报告里要声明** |

## SGLang 的 API 差异（踩过的坑）
- `generate(prompt, sampling_params, image_data=...)`，**不是** vLLM 的
  `{"prompt":..., "multi_modal_data":{"video":...}}`；
- **视频走 `video_data=`**，不是 `image_data=`（vLLM 侧我们传 `{"video": frames}`）；
- 命中数在 `out["meta_info"]["cached_tokens"]`。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# 路径锚点：复用主实验的 shared。
# 本文件在 vlm_inference_benchmark/sglang/ 下，所以 parents[1] = vlm_inference_benchmark/
# （注意：vllm/expN/ 下的脚本要多一层，是 parents[2] —— 这里少一层）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.common import MODEL_AWQ, write_jsonl  # noqa: E402


def build_prompt(question: str) -> str:
    """Qwen2-VL 模板 + **视频**占位符，与 vLLM 版逐字一致以保证可比。"""
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
                    default=Path(__file__).resolve().parents[1] /
                            "vllm" / "exp3_prefix_caching" / "workload.json",
                    help="与 vLLM 版**共用同一份** workload")
    ap.add_argument("--out", required=True, help="逐条结果 JSONL")
    ap.add_argument("--model", default=MODEL_AWQ)
    ap.add_argument("--num-questions", type=int, default=10)
    ap.add_argument("--frames", type=int, default=0, help="0 = 用 workload 里的帧数")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    # 显存：mfs 的语义与 vLLM 的 gmu **相反**（见 kv_cache_configurator.py:2165）
    ap.add_argument("--mem-fraction-static", type=float, default=0.90,
                    help="SGLang 的「留给非 KV 的比例」。越大 → slack 越小 → KV 越大。"
                         "实测 mfs=0.90 → KV 池 14,274 token")
    ap.add_argument("--context-length", type=int, default=4096,
                    help="exp3 最大前缀 6,448 token，故需 ≥8192；但 4096 用于中小帧数")
    ap.add_argument("--cuda-graph-max-bs-decode", type=int, default=2,
                    help="默认捕获 bs=[1,2,4,8] 实测耗时 337s；设 2 降到 3~5s 且省显存")
    ap.add_argument("--attention-backend", default="triton",
                    help="绕开 flashinfer JIT（本机系统 nvcc 12.0 不支持其编译选项）")
    ap.add_argument("--quantization", default="awq_marlin")
    ap.add_argument("--pixels-per-frame", type=int, default=262144,
                    help="每帧视觉像素预算，**必须与 vLLM 侧的 max_pixels 一致**才可比。"
                         "默认 262144（= vLLM exp3 用的值）。"
                         "SGLang 默认对视频用 602,112，会让同一张图产生 2.9× 的 token")
    ap.add_argument("--min-pixels", type=int, default=3136,
                    help="与 vLLM 侧 min_pixels 一致（SGLang 视频默认下限是 100,352）")
    ap.add_argument("--disable-radix-cache", action="store_true",
                    help="关掉 radix cache 作为对照（对应 vLLM 的 --no-enable-prefix-caching）")
    args = ap.parse_args()

    from PIL import Image
    import sglang as sgl

    spec = json.loads(args.workload.read_text(encoding="utf-8"))
    n_frames = args.frames or spec["frames"]

    # ⚠️ 口径对齐：视觉像素预算（本移植最费解的一处）
    #
    # 【问题】同一张图（1080×675）、同样 8 帧，两边 prompt token 数差 2.9×：
    #     vLLM   max_pixels=262,144        → 1,296 token  ← exp3 实测值
    #     SGLang 默认                       → 3,764 token
    # 差异**全部来自分辨率预算**，不是引擎机制 —— 不统一就是比错东西。
    #
    # 【踩坑记录】我先试了环境变量 `VIDEO_MAX_PIXELS`，**无效**。原因（读源码）：
    #   `qwen_vl.py:69` 的 `VIDEO_TOTAL_PIXELS` 在**模块导入时**求值一次，
    #   而真正生效的是 `preprocess_video(video, video_config=self.video_config)`
    #   （`qwen_vl.py:1027`），`self.video_config` 来自
    #   `base_processor.py:269`：`mm_process_config.get("video", {})`
    #   —— 即 **server 参数 `mm_process_config` 会覆盖模块常量**，模块常量只是默认值。
    #
    # 【正解】走 `Engine(mm_process_config={"video": {...}})`：
    #   `arg_groups/fields/mm.py:70` 定义 `mm_process_config: Optional[Dict[str, Any]]`，
    #   文档写「a json config contains keys: image, video, audio」。
    #   在 video 子字典里直接给 max_pixels/min_pixels，等价于 vLLM 的 mm_processor_kwargs。
    pixels_per_frame = args.pixels_per_frame          # 默认 262144，与 vLLM 的 max_pixels 一致

    # workload.json 里存的是相对仓库根的路径（可移植）
    from shared.common import INFRA_ROOT
    img = Path(spec["image"])
    image_path = img if img.is_absolute() else INFRA_ROOT / img
    image = Image.open(image_path).convert("RGB")
    # 【嵌套层级】SGLang 的多模态输入按「请求」分组：
    #   [单图/单视频]              → 一个请求，一个媒体
    #   [[图1, 图2, ...]]         → 一个请求，多个媒体
    #   [[...], [...]]            → 多个请求
    # 我们要的是「**一个请求、一段 16 帧的视频**」，所以必须是 [[f1, f2, ..., f16]]。
    # 直接传 [img]*16 会被理解成 16 个请求各一段单帧视频 → 报
    #   ValueError: Unsupported video input type: <class 'PIL.Image.Image'>
    # 这与 vLLM 不同：vLLM 的 {"video": [f1..f16]} 天然就是"一段视频"。
    video = [[image] * n_frames]
    questions = spec["questions"][: args.num_questions]

    tag = "off" if args.disable_radix_cache else "on"
    print(f"prefix cache(radix) = {tag}   N = {len(questions)}   帧数 = {n_frames}")
    print(f"mem_fraction_static = {args.mem_fraction_static}  context_length = {args.context_length}")
    print(f"每帧像素预算 = {pixels_per_frame:,}（min={args.min_pixels:,}）"
          f"  ← 与 vLLM 的 mm_processor_kwargs 对齐")

    kwargs = dict(
        model_path=args.model,
        quantization=args.quantization,
        attention_backend=args.attention_backend,
        mem_fraction_static=args.mem_fraction_static,
        context_length=args.context_length,
        cuda_graph_max_bs_decode=args.cuda_graph_max_bs_decode,
        # ★ 口径对齐：把视频每帧像素预算压到与 vLLM 相同的值
        mm_process_config={
            "video": {"max_pixels": pixels_per_frame, "min_pixels": args.min_pixels},
        },
        trust_remote_code=True,
        log_level="warning",
    )
    if args.disable_radix_cache:
        kwargs["disable_radix_cache"] = True

    llm = sgl.Engine(**kwargs)
    sampling = {"temperature": 0.0, "max_new_tokens": args.max_new_tokens}

    print(f"{'#':>3} {'prompt':>7} {'cached':>7} {'命中%':>7} {'wall s':>8} {'out_tok':>8}")
    print("-" * 50)

    records = []
    t_run0 = time.perf_counter()
    for i, q in enumerate(questions):
        prompt = build_prompt(q)
        t0 = time.perf_counter()
        # 【API 差异】视频走 video_data=；返回值是 dict，命中数在 meta_info 里
        out = llm.generate(prompt, sampling, video_data=video)
        wall = time.perf_counter() - t0

        meta = out.get("meta_info", {}) if isinstance(out, dict) else {}
        n_prompt = meta.get("prompt_tokens", 0)
        n_cached = meta.get("cached_tokens", 0)
        text = out.get("text", "") if isinstance(out, dict) else str(out)

        rec = {
            "i": i,
            "question": q,
            "prompt_tokens": n_prompt,
            "cached_tokens": n_cached,
            "hit_rate": round(n_cached / n_prompt, 4) if n_prompt else 0.0,
            "wall_s": round(wall, 4),
            "output_tokens": meta.get("completion_tokens", 0),
            "text_head": text[:60],
        }
        records.append(rec)
        print(f"{i:>3} {n_prompt:>7} {n_cached:>7} {rec['hit_rate'] * 100:>6.1f}% "
              f"{rec['wall_s']:>8.3f} {rec['output_tokens']:>8}")

    wall_total = time.perf_counter() - t_run0

    # 与 vLLM 版一致：剔除第 0 条（冷启动 + 一次性开销），报中位数
    warm = records[1:] or records
    cold = records[0]

    def med(key, rows):
        return st.median([r[key] for r in rows]) if rows else 0.0

    summary = {
        "engine": "sglang",
        "version": sgl.__version__,
        "prefix_caching": tag,
        "n_questions": len(questions),
        "frames": n_frames,
        "mem_fraction_static": args.mem_fraction_static,
        "context_length": args.context_length,
        "attention_backend": args.attention_backend,
        "quantization": args.quantization,
        "wall_time_s": round(wall_total, 2),
        "cold": {"wall_s": cold["wall_s"], "prompt_tokens": cold["prompt_tokens"]},
        "warm": {
            "mean_wall_s": round(st.mean([r["wall_s"] for r in warm]), 4),
            "median_wall_s": round(med("wall_s", warm), 4),
            "min_wall_s": round(min(r["wall_s"] for r in warm), 4),
            "max_wall_s": round(max(r["wall_s"] for r in warm), 4),
            "mean_hit_rate": round(st.mean([r["hit_rate"] for r in warm]), 4),
            "median_cached_tokens": round(med("cached_tokens", warm), 1),
            "median_prompt_tokens": round(med("prompt_tokens", warm), 1),
        },
    }

    out_path = Path(args.out)
    write_jsonl(out_path, summary, records)

    w = summary["warm"]
    print()
    print("=== 汇总（剔除以 0 条，与 vLLM 版口径一致）===")
    print(f"  冷启动        wall {cold['wall_s']:.3f}s")
    print(f"  后续请求      wall 中位 {w['median_wall_s']:.3f}s  "
          f"（范围 {w['min_wall_s']:.3f}~{w['max_wall_s']:.3f}）")
    print(f"  平均命中率    {w['mean_hit_rate'] * 100:.1f}%"
          f"（{w['median_cached_tokens']:.0f} / {w['median_prompt_tokens']:.0f} token）")
    print(f"  整轮墙钟      {wall_total:.1f}s")
    print(f"逐条结果：{out_path}")
    print("SGLANG_RUN_OK")

    llm.shutdown()


if __name__ == "__main__":
    main()
