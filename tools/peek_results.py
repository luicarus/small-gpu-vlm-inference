"""通用工具：查看实验结果逐条明细（人工验货用）。

用法：
    python tools/peek_results.py results/vlm_inference_benchmark/exp2_engine_comparison/hf_nf4_b1.jsonl [--n 5] [--wrong]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path)
    ap.add_argument("--n", type=int, default=0, help="只看前 N 条（0=全部）")
    ap.add_argument("--correct", action="store_true", help="只显示答对的")
    ap.add_argument("--wrong", action="store_true", help="只显示答错的")
    args = ap.parse_args()

    records = []
    meta = None
    with open(args.path, encoding="utf-8") as fp:
        for line in fp:
            obj = json.loads(line)
            if "_meta" in obj:
                meta = obj["_meta"]
            else:
                records.append(obj)

    if meta:
        print("=== 元信息 ===")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print()

    def hit(rec: dict) -> bool:
        # 命中判定：标准化后与任一真值相等，或真值被包含在输出里
        out = rec["output_norm"]
        return any(out == str(a).strip().lower() or str(a).strip().lower() in out
                   for a in rec["answers"])

    shown = 0
    n_ok = sum(1 for r in records if hit(r))
    for r in records:
        ok = hit(r)
        if args.correct and not ok:
            continue
        if args.wrong and ok:
            continue
        print(f"[{'✓' if ok else '✗'}] {r['id']}  ({r['question_type']})  {r['latency_s']}s")
        print(f"    Q: {r['question']}")
        print(f"    真值: {r['answers']}")
        print(f"    输出: {r['output']!r}")
        if r.get("error"):
            print(f"    ERROR: {r['error']}")
        print()
        shown += 1
        if args.n and shown >= args.n:
            break

    print(f"共 {len(records)} 条，命中 {n_ok} 条（{n_ok/len(records)*100:.1f}%）" if records else "无记录")


if __name__ == "__main__":
    main()
