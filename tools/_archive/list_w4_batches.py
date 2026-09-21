#!/usr/bin/env python3
"""按批次列出所有 JSONL 记录，用于定位周记录表格的取值来源。"""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results" / "vlm_inference_benchmark" / "vllm" / "preliminary_3b_memory_sweep"

for jf in sorted(RESULTS.glob("w4_sweep_*.jsonl")):
    print(f"===== {jf.name} =====")
    for line in jf.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        print(
            "  %-24s base=%-4s peak=%-5s delta=%-5s kv=%-5s tps=%-6s wall=%-6s %s"
            % (
                r["label"],
                r.get("base_mib", "—"),
                r.get("peak_mib", "—"),
                r.get("delta_mib", "—"),
                r.get("kv_cache_gib", "—"),
                r.get("tokens_per_s", "—"),
                r.get("wall_time_s", "—"),
                r.get("death", ""),
            )
        )
    print()

# 结论：哪些配置在哪个批次里"成功"，用于确定 §3.1 应引用哪个批次
print("=" * 70)
print("关键问题：08_offload2 在 195359 批次是 OK(peak=4017, tps=1.81)")
print("          在 201925 批次是 OK(peak=4045, tps=2.04)")
print("          周记录 §3.1 混用了：peak=4017(旧批次) + tps=2.04(新批次)")
print("=" * 70)
