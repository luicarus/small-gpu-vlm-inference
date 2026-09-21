"""汇总前置实验的全部扫描结果：把多个 w4_sweep_*.jsonl 合并打印成一张总表。

设计约定：**结果只保留一份文档**（原始数据在 results/，分析只打印不落盘），
所以本脚本只把总表打到屏幕，不生成新的 .md 文件——需要更新文档时手动粘贴即可。

为什么要单独做汇总脚本：
1. 扫描分多次运行（含复现验证），单次 summary 只有那一次的配置；
2. 同一 label 可能有多次运行记录（如 11/12 号先失败后成功），需要标明是哪一次；
3. 交付物需要一张「含峰值 / tok/s / KV 池 / 死法」的完整表，人工抄容易错。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 结果目录走 shared 的镜像规则（与 sweep.py 一致，避免手写层级）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.common import results_dir_of  # noqa: E402


RESULTS_DIR = results_dir_of(__file__)

# 列名 → 从记录里取值的键（None 表示需要特殊处理）
COLUMNS = [
    ("配置", "label"),
    ("gmu", None),
    ("offload", None),
    ("len", None),
    ("seqs", None),
    ("batched", None),
    ("跑前基线 MiB", "base_before_mib"),
    ("状态", "status"),
    ("死法", "death"),
    ("峰值 MiB", "peak_mib"),
    ("净增 MiB", "delta_mib"),
    ("KV 池 GiB", "kv_cache_gib"),
    ("KV tokens", "kv_cache_tokens"),
    ("并发", "max_concurrency"),
    ("tok/s", "tokens_per_s"),
    ("生成耗时 s", "gen_time_s"),
    ("输出 token", "output_tokens"),
    ("墙钟 s", "wall_time_s"),
    ("offload 实搬 GiB", "cpu_offloaded_gib"),
    ("显存内权重 GiB", "model_load_gib"),
]


def load_records() -> list[dict]:
    records: list[dict] = []
    for path in sorted(RESULTS_DIR.glob("w4_sweep_*.jsonl")):
        run_id = path.stem.replace("w4_sweep_", "")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_run"] = run_id
            records.append(record)
    return records


def cell(record: dict, key: str | None) -> str:
    if key is None:
        return ""
    value = record.get(key)
    if value is None:
        return "—"
    if key == "status" and value == "OK":
        return "✅ OK"
    if key == "status" and value == "FAIL":
        return "❌ FAIL"
    return str(value)


def param_cell(record: dict, name: str) -> str:
    params = record.get("params", {})
    mapping = {
        "gmu": "gpu_memory_utilization",
        "offload": "cpu_offload_gb",
        "len": "max_model_len",
        "seqs": "max_num_seqs",
        "batched": "max_num_batched_tokens",
    }
    value = params.get(mapping[name])
    if name == "batched" and not value:
        return "auto"
    return "—" if value is None else str(value)


def build_table(records: list[dict]) -> str:
    header = "| " + " | ".join(name for name, _ in COLUMNS) + " | 运行批次 |"
    sep = "|" + "---|" * (len(COLUMNS) + 1)
    rows = [header, sep]
    for record in records:
        cells = []
        for name, key in COLUMNS:
            cells.append(param_cell(record, name) if key is None else cell(record, key))
        cells.append(record["_run"][-6:])  # 批次时间戳后 6 位，便于追溯
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    records = load_records()
    if not records:
        print(f"没有找到扫描记录：{RESULTS_DIR}/w4_sweep_*.jsonl")
        return

    ok = [r for r in records if r.get("status") == "OK"]
    fail = [r for r in records if r.get("status") != "OK"]

    table = build_table(records)
    print("# 前置实验扫描结果总表（全部运行合并）")
    print()
    print(f"来源：`results/preliminary_3b_memory_sweep/w4_sweep_*.jsonl`，共 {len(records)} 条记录"
          f"（OK {len(ok)} / FAIL {len(fail)}）")
    print()
    print(table)
    print()
    print("## 失败配置一览")
    print()
    print("| 配置 | 死法 | 日志尾部 |")
    print("|---|---|---|")
    for record in fail:
        death = record.get("death", "unknown")
        print(f"| {record['label']} | {death} | {record.get('log_tail', '—')[:80]} |")
    print()
    print("→ 结果只保留一份文档：把上表粘进报告即可，不再另建 summary 文件")


if __name__ == "__main__":
    main()
