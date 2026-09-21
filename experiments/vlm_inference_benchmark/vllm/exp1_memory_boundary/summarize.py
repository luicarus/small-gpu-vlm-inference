"""exp1 汇总：把扫描结果整理成能直接写进报告的表。

只打印，不落盘 —— 全项目统一约定：分析脚本的输出由人按需粘进文档，
避免"脚本自动改文档"导致数据与叙述不一致（本项目统一遵守这条）。
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.common import RESULTS_ROOT, results_dir_of, read_jsonl  # noqa: E402

RESULTS_DIR = results_dir_of(__file__)

DEATH_LABEL = {
    "ok": "—",
    "budget": "#1 预算越界",
    "no_kv": "#2 KV 池为空",
    "seq_too_long": "#3 序列太长",
    "oom_runtime": "运行期 OOM",
    "timeout": "超时",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    files = sorted(glob.glob(str(args.results_dir / "exp1_sweep_*.jsonl")))
    if not files:
        print(f"没找到结果：{args.results_dir}/exp1_sweep_*.jsonl")
        return
    latest = files[-1]
    meta, rows = read_jsonl(Path(latest))

    print(f"# exp1 显存边界扫描（模型：{meta['model']}）")
    print(f"\n来源：`{Path(latest).name}`，共 {meta['n_configs']} 个配置"
          f"（OK {meta['n_ok']} / FAIL {meta['n_fail']}）\n")

    print("## 总表\n")
    print("| 配置 | gmu | offload | max_len | seqs | mnbt | 状态 | 死法 | "
          "峰值 MiB | KV GiB | KV tokens | 并发上限 | non_kv MiB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['label']} | {r['gpu_memory_utilization']} | {r['cpu_offload_gb']} | "
              f"{r['max_model_len']} | {r['max_num_seqs']} | {r['max_num_batched_tokens']} | "
              f"{r['status']} | {DEATH_LABEL.get(r['death'], r['death'])} | "
              f"{r.get('peak_mib', 0)} | {r.get('kv_gib', 0):.2f} | "
              f"{r.get('kv_tokens', 0):,} | {r.get('conc_x', 0):.2f}x | "
              f"{r.get('non_kv_mib', 0)} |")

    # 从 OK 的配置里推 KV 池的可用带宽与内存账本
    ok = [r for r in rows if r["status"] == "OK" and r.get("kv_tokens")]
    if ok:
        print("\n## 关键量（取 OK 配置的实测值）\n")
        non_kv = [r["non_kv_mib"] for r in ok if r.get("non_kv_mib")]
        if non_kv:
            print(f"- **non_kv（权重 + 激活 + CUDA graph）**："
                  f"{min(non_kv)} ~ {max(non_kv)} MiB（中位 {sorted(non_kv)[len(non_kv)//2]}）")
        kv_gib = [r["kv_gib"] for r in ok]
        print(f"- **KV 池**：{min(kv_gib):.2f} ~ {max(kv_gib):.2f} GiB")
        # KV 每 token 成本 = 池大小 / token 数
        per_tok = [(r["kv_gib"] * 1024 * 1024) / r["kv_tokens"] for r in ok if r["kv_tokens"]]
        if per_tok:
            print(f"- **KV 成本**：{min(per_tok):.1f} ~ {max(per_tok):.1f} KiB/token")

    fail = [r for r in rows if r["status"] != "OK"]
    if fail:
        print("\n## 失败配置与死法\n")
        for r in fail:
            head = f"- **{r['label']}**（{DEATH_LABEL.get(r['death'], r['death'])}）"
            err = (r.get("error") or "").strip()
            print(f"{head}：`{err[:150]}`" if err else head)


if __name__ == "__main__":
    main()