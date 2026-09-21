"""从 OCRBench 抽 100 条小图子集（可复现的确定性抽样）。

为什么只取小图任务：
1. **分辨率是最大的混杂变量**。OCRBench 文档类任务图片可达 970 KiB，要读出字必须开大
   分辨率 → 视觉 token 暴涨 → 4GB 卡上显存告急；而小图任务（1~47 KiB）原生分辨率低，
   不需要高 max_pixels，两边引擎的视觉 token 数就可控且一致。
2. **评分干净**。识别类任务答案是单个词/数字（exact match 即可判定），
   不需要 ANLS/框坐标等复杂指标，减少"评分器本身出错"的风险。
3. **对量化差异敏感**。细粒度字符识别正是 AWQ(激活感知) 与 NF4(信息论最优) 路线
   差异最容易暴露的地方。

抽样是**确定性**的（固定 seed），保证任何人复现都得到同一批 100 条——
benchmark 报告里"数据可复现"是硬要求。
"""

from __future__ import annotations

import argparse
import base64
import json
import random
from collections import Counter
from pathlib import Path

# 小图任务白名单：6 类字符识别（各 50 条）+ 场景文本 VQA（200 条）
SMALL_TASKS = [
    "Regular Text Recognition",
    "Irregular Text Recognition",
    "Artistic Text Recognition",
    "Handwriting Recognition",
    "Digit String Recognition",
    "Non-Semantic Text Recognition",
    "Scene Text-centric VQA",
]

# 分层配额：识别类每类 10 条（60）+ 场景文本 VQA 40 条 = 100
QUOTA = {t: 10 for t in SMALL_TASKS if t != "Scene Text-centric VQA"}
QUOTA["Scene Text-centric VQA"] = 40


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default=str(Path.home() / "datasets/OCRBench/data/test-00000-of-00001.parquet"))
    parser.add_argument("--out-dir", default=str(Path.home() / "datasets/OCRBench/subset100"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-type", type=int, default=0,
                        help="覆盖默认配额：每类只取 N 条（冒烟用）")
    args = parser.parse_args()

    import pyarrow.parquet as pq

    rows = pq.ParquetFile(args.parquet).read().to_pylist()
    rng = random.Random(args.seed)

    # 按任务类型分桶后各自随机抽样——直接随机抽 100 条会因分布倾斜而选不中稀缺类别
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        if r["question_type"] in SMALL_TASKS:
            buckets.setdefault(r["question_type"], []).append(r)

    picked: list[dict] = []
    for task in SMALL_TASKS:
        pool = buckets.get(task, [])
        rng.shuffle(pool)
        n = args.limit_per_type or QUOTA[task]
        picked.extend(pool[:n])

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for i, r in enumerate(picked):
        raw = r["image"]["bytes"]
        if not raw:
            continue
        item_id = f"{i:03d}_{r['dataset']}"
        # 图片落盘而非 base64 进 JSONL：便于人工核对、也便于脚本按需读单张
        img_path = img_dir / f"{item_id}.jpg"
        img_path.write_bytes(raw)
        items.append({
            "id": item_id,
            "dataset": r["dataset"],
            "question_type": r["question_type"],
            "question": r["question"],
            "answers": r["answer"],          # 真值列表（OCRBench 允许多个等价答案）
            "image": str(img_path),
            "image_kib": round(len(raw) / 1024, 1),
        })

    items_path = out_dir / "items.jsonl"
    with open(items_path, "w", encoding="utf-8") as fp:
        for it in items:
            fp.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"子集已生成：{items_path}")
    print(f"条数：{len(items)}   seed={args.seed}")
    print("\n=== 任务类型分布 ===")
    for k, v in Counter(it["question_type"] for it in items).most_common():
        print(f"  {v:4d}  {k}")
    sizes = [it["image_kib"] for it in items]
    print(f"\n图片体积：最小 {min(sizes)} KiB / 中位 {sorted(sizes)[len(sizes)//2]} KiB / 最大 {max(sizes)} KiB")


if __name__ == "__main__":
    main()
