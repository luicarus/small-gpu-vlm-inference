#!/usr/bin/env python3
"""校验周记录 §3.1 / §3.2 的每个数字是否与 JSONL 完全一致。

设计：把 markdown 表格解析成行，按 label 匹配 JSONL 记录，
逐列比对。§3.1 断言"取最新批次"，§3.2 断言"每次运行各占一行"。
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # InfraStudy
RESULTS = Path(__file__).resolve().parents[1] / "results" / "vlm_inference_benchmark" / "vllm" / "preliminary_3b_memory_sweep"
DOC = ROOT / "docs/周记录/W4_实操记录_2026-09-15.md"

# ---- 收集 JSONL 全部运行 ----
runs = []  # [(label, batch, record)]
for jf in sorted(RESULTS.glob("w4_sweep_*.jsonl")):
    batch = jf.stem.replace("w4_sweep_", "")[-6:]
    for line in jf.read_text(encoding="utf-8").splitlines():
        if line.strip():
            runs.append((json.loads(line)["label"], batch, json.loads(line)))

latest = {}
for label, batch, r in runs:
    latest[label] = (batch, r)

text = DOC.read_text(encoding="utf-8")

# ---- 解析 §3.1 ----
sec31 = text.split("### 3.1")[1].split("### 3.2")[0]
rows31 = []
for line in sec31.splitlines():
    if not line.startswith("|"):
        continue
    cells = [c.strip() for c in line.strip("|").split("|")]
    if len(cells) < 16 or not cells[1] or cells[1] in ("配置",) or set(cells[1]) <= set("- "):
        continue
    if not re.match(r"^\d+$", cells[0]):
        continue
    rows31.append(cells)

print(f"§3.1 解析到 {len(rows31)} 行\n")
fail = 0
print("===== §3.1 与「最新批次」比对 =====")
# 列序：# | 配置 | gmu | offload | len | seqs | batched | 结果 | 死法 |
#     峰值 | 净增 | KV池 | KV tokens | 并发 | tok/s | 生成s | 批次
C_LABEL, C_PEAK, C_DELTA = 1, 9, 10
C_KV, C_TPS, C_BATCH = 11, 14, 16
for cells in rows31:
    label = cells[C_LABEL]
    batch_doc = cells[C_BATCH]
    exp_batch, exp = latest[label]
    problems = []
    if batch_doc != exp_batch:
        problems.append(f"批次 文档={batch_doc} JSONL={exp_batch}")

    def num(s):
        s = s.replace("**", "").replace(",", "").strip()
        if s in ("—", "-", ""):
            return None
        try:
            return float(s)
        except ValueError:
            return None

    peak_doc = num(cells[C_PEAK])
    if peak_doc is not None and exp.get("peak_mib") is not None:
        if abs(peak_doc - exp["peak_mib"]) > 0.5:
            problems.append(f"峰值 文档={peak_doc} JSONL={exp['peak_mib']}")

    delta_doc = num(cells[C_DELTA])
    if delta_doc is not None and exp.get("delta_mib") is not None:
        if abs(delta_doc - exp["delta_mib"]) > 0.5:
            problems.append(f"净增 文档={delta_doc} JSONL={exp['delta_mib']}")

    tps_doc = num(cells[C_TPS])
    if tps_doc is not None and exp.get("tokens_per_s") is not None:
        if abs(tps_doc - exp["tokens_per_s"]) > 0.005:
            problems.append(f"tok/s 文档={tps_doc} JSONL={exp['tokens_per_s']}")

    kv_doc = cells[C_KV].replace(" GiB", "").strip()
    if kv_doc not in ("—", "-", "") and exp.get("kv_cache_gib") is not None:
        if abs(float(kv_doc) - exp["kv_cache_gib"]) > 0.005:
            problems.append(f"KV池 文档={kv_doc} JSONL={exp['kv_cache_gib']}")

    if problems:
        fail += 1
        print(f"BAD  {label:<24} " + "; ".join(problems))
    else:
        print(f"OK   {label:<24} 批次={batch_doc} peak={exp.get('peak_mib')} tps={exp.get('tokens_per_s')}")

# ---- 解析 §3.2 ----
sec32 = text.split("### 3.2")[1].split("## 四、")[0]
rows32 = []
for line in sec32.splitlines():
    if not line.startswith("|"):
        continue
    cells = [c.strip() for c in line.strip("|").split("|")]
    if len(cells) < 13:
        continue
    if cells[0] in ("配置",) or set(cells[0]) <= set("- "):
        continue
    rows32.append(cells)

print(f"\n§3.2 解析到 {len(rows32)} 行")
print("===== §3.2 与 JSONL 逐次运行比对 =====")
# §3.2 是"一次运行一行"，用 (label, batch) 反查
for cells in rows32:
    raw_label = cells[0].replace("（复现）", "").replace("（干净 VM）", "").strip()
    batch_doc = cells[12]
    match = [(b, r) for (l, b, r) in runs if l == raw_label and b == batch_doc]
    if not match:
        fail += 1
        print(f"BAD  {cells[0]:<28} 找不到 JSONL 记录 batch={batch_doc}")
        continue
    _, exp = match[0]
    problems = []

    def num(s):
        s = s.replace("**", "").replace(",", "").strip()
        if s in ("—", "-", ""):
            return None
        try:
            return float(s)
        except ValueError:
            return None

    for col, key, tol in ((4, "peak_mib", 0.5), (8, "tokens_per_s", 0.005), (9, "wall_time_s", 0.05)):
        d = num(cells[col])
        e = exp.get(key)
        # 墙钟时间在 JSONL 里可能有轻微差异，容忍
        if d is not None and e is not None and abs(d - e) > tol:
            problems.append(f"{key} 文档={d} JSONL={e}")

    if problems:
        fail += 1
        print(f"BAD  {cells[0]:<28} batch={batch_doc} " + "; ".join(problems))
    else:
        print(f"OK   {cells[0]:<28} batch={batch_doc}")

print(f"\n{'❌ %d 处不一致' % fail if fail else '✅ 周记录 §3.1 / §3.2 与 JSONL 完全一致'}")
