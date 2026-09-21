"""四指标对比分析 —— 读取各引擎逐条结果，算出可写进报告的对比表。

同时支持 2B 正式版（Qwen2-VL-2B）与前置的 3B 实验——只要传 `--results-dir` 即可，
两边的 JSONL 结构完全一致（同一套执行器产出）。

指标定义（口径必须写死，否则数字没法复核）：
1. **吞吐**：items/s 与 out_tokens/s。token 数用 Qwen 分词器对输出重新编码得到——
   两个引擎共用同一分词器，所以口径一致（比"按字符数估算"严谨）。
2. **显存峰值**：NVML 整卡 memory.used 峰值，同时给出基线以便看净增。
3. **延迟**：单条延迟的 P50 / P90 / P99（batch>1 时是批内均摊值，表头会标注）。
4. **答案一致率 + 准确率**：
   - 准确率 = 标准化后命中真值（含包含判定）的比例，按引擎分别算；
   - 一致率 = 两个引擎在**同一条**上输出是否等价，回答"换引擎会不会改变答案"。
   ⚠️ 统计功效：官方数据表明 AWQ 相对 BF16 只掉 1~3 分，100 条样本的分辨力约 ±5 分，
      因此准确率差异**不可**解读为量化优劣结论，只能作为"没有灾难性退化"的回归检查。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# 路径锚点走 shared，避免固定层级（脚本被搬过位置）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.common import RESULTS_ROOT, results_dir_of  # noqa: E402


def load_run(path: Path) -> tuple[dict, list[dict]]:
    meta, records = None, []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            obj = json.loads(line)
            if "_meta" in obj:
                meta = obj["_meta"]
            else:
                records.append(obj)
    return meta or {}, records


def norm_hit(rec: dict) -> bool:
    out = rec["output_norm"]
    for a in rec["answers"]:
        a = str(a).strip().lower()
        if out == a or (a and a in out):
            return True
    return False


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q / 100 * (len(s) - 1)))))
    return s[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path,
                    default=results_dir_of(__file__))
    ap.add_argument("--tokenizer", default=str(Path.home() / "models/Qwen2-VL-2B-Instruct"),
                    help="用于把输出文本重新编码成 token 数（两引擎共用同一分词器，口径才一致）")
    args = ap.parse_args()

    runs = {}
    # 覆盖全部 batch 档位（1/2/4/8），保证四指标表与批曲线取自同一批运行
    tags = [f"{eng}_{q}_b{b}"
            for b in ("1", "2", "4", "8")
            for eng, q in (("hf", "nf4"), ("vllm", "awq"))]
    for tag in tags:
        path = args.results_dir / f"{tag}.jsonl"
        if path.exists():
            runs[tag] = load_run(path)
        else:
            print(f"[跳过] 未找到 {path}")

    if not runs:
        print("没有可分析的结果文件")
        return

    # 输出 token 数：用分词器重编码输出文本（两引擎同分词器，口径一致）
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def out_tokens(texts: list[str]) -> int:
        return sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts if t)

    print("## 1. 四指标对比\n")
    print("| 配置 | 条数 | 运行时长 s | 吞吐 items/s | 输出 token 数 | 吞吐 tok/s | 显存基线 MiB | **显存峰值 MiB** | 净增 MiB | 加载 s | 总墙钟 s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    summary = {}
    for tag, (meta, recs) in runs.items():
        texts = [r["output"] for r in recs]
        ntokens = out_tokens(texts)
        run_t = meta.get("run_time_s") or 0
        items_s = len(recs) / run_t if run_t else 0
        tok_s = ntokens / run_t if run_t else 0
        base, peak = meta.get("base_mib"), meta.get("peak_mib")
        delta = (peak - base) if (peak and base) else None
        summary[tag] = dict(meta=meta, recs=recs, items_s=items_s, tok_s=tok_s,
                            ntokens=ntokens, delta=delta)
        print(f"| {tag} | {len(recs)} | {run_t} | {items_s:.2f} | {ntokens} | {tok_s:.1f} | "
              f"{base} | **{peak}** | {delta} | {meta.get('load_time_s')} | {meta.get('wall_time_s')} |")

    print("\n## 2. 单条延迟分位（秒）\n")
    print("| 配置 | P50 | P90 | P99 | 均值 | 最大 |")
    print("|---|---|---|---|---|---|")
    for tag, s in summary.items():
        lat = [r["latency_s"] for r in s["recs"]]
        print(f"| {tag} | {percentile(lat,50):.3f} | {percentile(lat,90):.3f} | "
              f"{percentile(lat,99):.3f} | {statistics.mean(lat):.3f} | {max(lat):.3f} |")

    print("\n## 3. 准确率（vs OCRBench 真值）\n")
    print("| 配置 | 命中 | 准确率 |")
    print("|---|---|---|")
    acc = {}
    for tag, s in summary.items():
        recs = s["recs"]
        hits = sum(1 for r in recs if norm_hit(r))
        acc[tag] = hits / len(recs) * 100 if recs else 0
        print(f"| {tag} | {hits}/{len(recs)} | {acc[tag]:.1f}% |")

    # 按任务类型拆开看——某些类别可能对量化更敏感，汇总数字会掩盖它
    print("\n### 3.1 分类准确率\n")
    types = sorted({r["question_type"] for s in summary.values() for r in s["recs"]})
    header = "| 任务类型 | " + " | ".join(summary.keys()) + " |"
    print(header)
    print("|" + "---|" * (len(summary) + 1))
    for t in types:
        cells = []
        for s in summary.values():
            sub = [r for r in s["recs"] if r["question_type"] == t]
            cells.append(f"{sum(1 for r in sub if norm_hit(r))}/{len(sub)}" if sub else "—")
        print(f"| {t} | " + " | ".join(cells) + " |")

    # 一致率只在 batch=1 的两个引擎之间算——batch 不同会影响输出，混着比没有意义
    b1 = [t for t in ("hf_nf4_b1", "vllm_awq_b1") if t in summary]
    if len(b1) == 2:
        a_tag, b_tag = b1
        a_map = {r["id"]: r for r in summary[a_tag]["recs"]}
        b_map = {r["id"]: r for r in summary[b_tag]["recs"]}
        common = sorted(set(a_map) & set(b_map))
        same = [i for i in common if a_map[i]["output_norm"] == b_map[i]["output_norm"]]
        print(f"\n## 4. 跨引擎答案一致率（{a_tag} vs {b_tag}，同条对比）\n")
        print(f"- 可比条数：**{len(common)}**")
        print(f"- 输出等价（标准化后完全相同）：**{len(same)}/{len(common)} = {len(same)/len(common)*100:.1f}%**")

        # 【关键修正】裸字符串一致率会低估一致性：一个引擎整句作答（"The text reads "X"."），
        # 另一个直接给词（"X"），答案其实一致却被判为不同。用双向包含做等价判定，
        # 剥离风格差异后再看真实分歧。两个数字都要报——它们回答的是不同问题：
        # 严格一致率（换引擎输出是否逐字相同） vs 语义一致率（答案是否相同）。
        def equivalent(x: str, y: str) -> bool:
            if not x or not y:
                return False
            return x == y or x in y or y in x

        sem_same = [i for i in common if equivalent(a_map[i]["output_norm"], b_map[i]["output_norm"])]
        print(f"- 语义等价（双向包含，剥离整句/短答风格差异）：**{len(sem_same)}/{len(common)} = {len(sem_same)/len(common)*100:.1f}%**")

        diff = [i for i in common if i not in set(sem_same)]
        both_wrong = [i for i in diff if not norm_hit(a_map[i]) and not norm_hit(b_map[i])]
        print(f"- 真实分歧条数：{len(diff)}（其中两个引擎都没命中真值：{len(both_wrong)}，"
              f"即「只有一方答对」的条数：{len(diff) - len(both_wrong)}）")
        print("\n### 4.1 不一致样例（最多 10 条）\n")
        for i in diff[:10]:
            ra, rb = a_map[i], b_map[i]
            print(f"- `{i}` ({ra['question_type']}) 真值={ra['answers']}")
            print(f"    - {a_tag}: {ra['output']!r}")
            print(f"    - {b_tag}: {rb['output']!r}")


if __name__ == "__main__":
    main()
