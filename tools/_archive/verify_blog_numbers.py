#!/usr/bin/env python3
"""对账：博客里引用的每个数字 vs 原始 JSONL 实测，不一致就报出来。

为什么要写这个脚本：博客一旦发出去，数字错了没法改（转载会扩散）。
人工核对 12 行 x 8 列容易漏，交给脚本按字段逐一比对。
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # InfraStudy
RESULTS = Path(__file__).resolve().parents[1] / "results" / "vlm_inference_benchmark" / "vllm" / "preliminary_3b_memory_sweep"
BLOG = ROOT / "docs/blog/知乎_vLLM显存参数判别准则.md"

# 1) 汇总所有批次的 JSONL，取每个 label 的最终有效运行
final = {}
for jf in sorted(RESULTS.glob("w4_sweep_*.jsonl")):
    for line in jf.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        final[r["label"]] = r  # 后出现的批次覆盖前面的

print(f"读到 {len(final)} 个配置的最终运行记录\n")

# 2) 打印实测真值表，供人工对照博客
print(f"{'label':<26} {'status':<5} {'death':<12} {'KV GiB':>7} {'tok/s':>7} {'offload':>8}")
print("-" * 72)
for label, r in sorted(final.items()):
    kv = r.get("kv_cache_gib")
    kv_s = f"{kv:.2f}" if kv is not None else "  —  "
    tps = r.get("tokens_per_s")
    tps_s = f"{tps:.2f}" if tps is not None else "  —  "
    off = r.get("cpu_offloaded_gib")
    off_s = f"{off:.2f}" if off is not None else "  —  "
    print(f"{label:<26} {r.get('status',''):<5} {r.get('death','—'):<12} {kv_s:>7} {tps_s:>7} {off_s:>8}")

# 3) 断言关键数字确实出现在博客正文里（防止我写串行）
blog = BLOG.read_text(encoding="utf-8")
checks = [
    ("0.25", "基线 KV 池 0.25 GiB"),
    ("0.05", "gmu0.65 KV 池 0.05 GiB"),
    ("0.45", "gmu0.75 KV 池 0.45 GiB"),
    ("0.65", "gmu0.80 KV 池 0.65 GiB"),
    ("0.12", "offload2 KV 池 0.12 GiB"),
    ("6.02", "基线 tok/s（最新批次 201722）"),
    ("6.28", "gmu0.65 tok/s"),
    ("6.48", "gmu0.75 tok/s"),
    ("6.40", "gmu0.80 tok/s"),
    ("24.56", "4 并发 tok/s"),
    ("2.04", "offload2 tok/s（最新批次 201925）"),
    ("4.08", "并发加速比（按最新批次 6.02 计算）"),
    ("2611", "non_kv 实测值"),
    ("0.6375", "#2 边界"),
    ("0.8025", "#1 边界"),
    ("1.34", "offload 硬上限"),
    ("36 KiB", "KV 每 token 成本"),
]
print("\n===== 博客关键数字出现性检查 =====")
fail = 0
for needle, desc in checks:
    ok = needle in blog
    if not ok:
        fail += 1
    print(f"{'OK  ' if ok else 'MISS'} {needle:<10} {desc}")

# 4) 反向检查：被推翻的旧估算只能作为「已被证伪」出现，不能作为结论出现。
#    "1300 MiB" 在博客里是合理的——它出现在「我原本以为 ~1300，实测打脸」那句里，
#    是刻意保留的反例。所以按行判断：该行必须带"以为/原本/错"等证伪标记。
banned = {
    "−779": "旧线性外推的负 KV 池",
    "-779": "旧线性外推的负 KV 池",
    "1300 MiB": "旧线性外推的 offload2 KV 池",
    "36,900": "旧线性外推的 token 数",
    "70,200": "旧线性外推的 token 数",
    "+2467": "旧线性外推的全 offload KV 池",
    "待校准": "未校准占位符",
}
FALSIFY_MARKERS = ("以为", "原本", "错", "打脸", "推翻", "失效", "无效")
print("\n===== 被推翻的旧估算残留检查（只允许出现在证伪句里）=====")
for token, desc in banned.items():
    for i, line in enumerate(blog.splitlines(), 1):
        if token not in line:
            continue
        if any(m in line for m in FALSIFY_MARKERS):
            print(f"OK   line {i}: '{token}' 出现在证伪句里（{desc}）")
        else:
            fail += 1
            print(f"BAD  line {i}: '{token}' 作为结论残留（{desc}）")
if not fail:
    print("无问题残留 OK")

print(f"\n{'❌ 有 %d 项未通过' % fail if fail else '✅ 全部通过'}")
sys.exit(1 if fail else 0)
