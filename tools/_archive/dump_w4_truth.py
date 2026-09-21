#!/usr/bin/env python3
"""以 JSONL 为唯一口径，导出每个配置的真值，用于对齐周记录。

背景：周记录是人工誊写的，与 JSONL 存在细微差异（6.01 vs 6.02、KV 池 0.11 vs 0.12）。
本脚本不做修改，只打印"JSONL 权威值"，供人工比对周记录后订正。
"""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results" / "vlm_inference_benchmark" / "vllm" / "preliminary_3b_memory_sweep"

# 同一 label 可能在多个批次出现（复现运行）。JSONL 按批次文件名排序，
# 后出现的批次覆盖前面的 —— 与 summarize.py 的取数逻辑保持一致：
# 每个 label 取"最终有效运行"（最新批次里的那条）。
final = {}
for jf in sorted(RESULTS.glob("w4_sweep_*.jsonl")):
    for line in jf.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        r["_batch"] = jf.stem.replace("w4_sweep_", "")
        final[r["label"]] = r

FIELDS = [
    ("status", "结果", "{}"),
    ("death", "死法", "{}"),
    ("base_mib", "跑前基线", "{}"),
    ("peak_mib", "峰值", "{}"),
    ("delta_mib", "净增", "{}"),
    ("kv_cache_gib", "KV池GiB", "{}"),
    ("kv_cache_tokens", "KV tokens", "{}"),
    ("max_concurrency", "并发", "{}"),
    ("tokens_per_s", "tok/s", "{}"),
    ("gen_time_s", "生成s", "{}"),
    ("cpu_offloaded_gib", "offload实搬", "{}"),
    ("model_load_gib", "显存内权重", "{}"),
    ("wall_time_s", "墙钟s", "{}"),
]

print(f"JSONL 权威值（共 {len(final)} 个配置）\n")
for label in sorted(final):
    r = final[label]
    print(f"### {label}   [批次 {r['_batch']}]")
    parts = []
    for key, name, _ in FIELDS:
        if key not in r:
            continue
        v = r[key]
        if isinstance(v, float):
            v = f"{v:.2f}"
        parts.append(f"{name}={v}")
    # 每行 5 个字段，便于阅读
    for i in range(0, len(parts), 5):
        print("   " + " | ".join(parts[i : i + 5]))
    print()

# 特别标注：哪些 label 有多个批次、值是否一致（复现性）
print("=" * 60)
print("多批次 label 的复现一致性检查")
print("=" * 60)
by_label = {}
for jf in sorted(RESULTS.glob("w4_sweep_*.jsonl")):
    for line in jf.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        by_label.setdefault(r["label"], []).append((jf.stem, r))

for label, runs in sorted(by_label.items()):
    if len(runs) < 2:
        continue
    print(f"\n{label}（{len(runs)} 次运行）")
    for batch, r in runs:
        kv = r.get("kv_cache_gib")
        tps = r.get("tokens_per_s")
        kv_s = f"{kv:.2f}" if kv is not None else "—"
        tps_s = f"{tps:.2f}" if tps is not None else "—"
        print(f"   {batch}  status={r.get('status'):<5} KV={kv_s:<6} tok/s={tps_s:<6}")
