"""生成「共享前缀」workload —— 同一段伪视频 + N 个不同问题。

## 为什么这么造
prefix caching 只在**多个请求共享同一个前缀**时才有收益。用 100 张各不相同的图
（如 OCRBench）时共享部分几乎为零，测不出收益（理论上限≈0）。

本 workload 复现真实业务形态：**一份大前缀（视频）+ 多个只问不同问题**。
- 共享前缀 = 同一张图重复 K 帧（伪视频）→ 视觉 token 数大、且完全一致
- 各请求只在小尾巴（问题文本）上不同 → 前缀的 KV 应被完整复用

## 为什么用「图片重复成帧」而不是真视频
① 不需要准备视频素材；② 帧完全一致 → 前缀哈希必然一致，变量最干净；
③ 与「多 QA 共享视频前缀」的业务形态同构。

## 问题集
- 前 24 条是手写的、针对**当前测试图内容**的真实问题（答案可判对错）
- 其余按模板生成，用于把 N 拉到 50，验证收益随 N 的伸缩

> ⚠️ 换测试图（`assets/test_image.png`）时**必须同步更新这 24 条问题**，
> 否则问题与图片内容不匹配，答案失去意义。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 路径锚点走 shared：不写死层级，脚本被搬位置也不会读错图。
# parents[2] = vlm_inference_benchmark/（本文件在 vllm/exp3_prefix_caching/ 下）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.common import DEFAULT_IMAGE  # noqa: E402

# 针对 assets/test_image.png（动漫人物插画）的手写问题
HANDWRITTEN = [
    "请描述这张图的主要内容和整体色调。",
    "图中人物的头发是什么颜色的？",
    "图中人物的眼睛是什么颜色？",
    "图中人物的服装是什么颜色？",
    "图中人物的表情大致是怎样的？",
    "图中人物头上或发间有什么装饰？",
    "图中人物手里或身前有什么？",
    "图片右侧的文字写的是什么？",
    "图片右下方有什么标志或文字？",
    "这张图的背景是偏亮还是偏暗？",
    "图中人物的头发是长发还是短发？",
    "画面的主体是一个人物还是多个人物？",
    "图中人物面向画面的哪一侧？",
    "这张图是照片还是插画？",
    "请说出图中出现的任意一个英文单词。",
    "图中人物颈部的服饰有什么特征？",
    "画面中较暗的区域大致在哪个位置？",
    "图中人物手中发光的部分是什么颜色？",
    "这张图的构图中，人物大致占据画面的哪个区域？",
    "图中人物发色与服装颜色哪个更浅？",
    "请用一句话概括这张图的氛围。",
    "图片中是否出现了心形图案？",
    "图中人物的发饰与服装颜色是否一致？",
    "如果要给这张图起一个标题，你会起什么？",
]

TEMPLATE = "请描述图中人物身上第 {k} 处细节。"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "workload.json"))
    ap.add_argument("--questions", type=int, default=50, help="问题总数（最多 24 手写 + 模板补齐）")
    ap.add_argument("--frames", type=int, default=16, help="伪视频帧数（同一张图重复）")
    ap.add_argument("--image", default=str(DEFAULT_IMAGE),
                    help="共享前缀用的图片（默认取 assets/ 下的测试图）")
    args = ap.parse_args()

    questions = list(HANDWRITTEN)
    k = 1
    while len(questions) < args.questions:
        questions.append(TEMPLATE.format(k=k))
        k += 1
    questions = questions[:args.questions]

    # 写**相对仓库根**的路径，而不是绝对路径：
    # 绝对路径只在本机有效，公开仓库里它既不必要也会泄露目录结构。
    from shared.common import INFRA_ROOT
    try:
        image_rel = str(Path(args.image).resolve().relative_to(INFRA_ROOT))
    except ValueError:
        image_rel = str(Path(args.image).resolve())      # 仓库外的图，保持绝对路径

    payload = {
        "image": image_rel,
        "frames": args.frames,
        "questions": questions,
        "note": "同一张图重复 frames 帧构成共享前缀；各请求只在小尾巴（问题）上不同",
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"workload 已生成：{out}")
    print(f"  图片   : {payload['image']}")
    print(f"  帧数   : {args.frames}（伪视频，同一张图重复）")
    print(f"  问题数 : {len(questions)}")
    print(f"  前 2 条: {questions[0]}")
    print(f"          {questions[1]}")


if __name__ == "__main__":
    main()
