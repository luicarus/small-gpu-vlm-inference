#!/usr/bin/env bash
# 检查 vLLM 日志里有没有「运行时 JIT 编译」警告 + 逐条延迟汇总
#
# 为什么需要它：Triton 遇到 warmup 未覆盖的 shape 会在**推理中途**编译 kernel，
# 造成秒级长尾。只有看分位数才发现得了（P50 往往完全正常）。
set -uo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../shared" && pwd)/paths.sh"
bench_init "${BASH_SOURCE[0]}"

R="$(bench_results_dir "${BASH_SOURCE[0]}")"
cd "$R/logs" || { echo "找不到日志目录 $R/logs" >&2; exit 1; }

echo "=== Triton 运行时 JIT 警告次数 ==="
for f in vllm_awq_b*.log; do
  [ -f "$f" ] || continue
  n=$(grep -c "Triton kernel JIT compilation during inference" "$f" 2>/dev/null || echo 0)
  printf "  %-18s %s 次\n" "$f" "$n"
done

echo
echo "=== 逐条延迟汇总（看有没有 10s 级长尾）==="
cd "$R" || exit 1
"$(bench_python)" - <<'PYEOF'
import json, os
print(f"  {'配置':<20}{'总推理 s':>10}{'中位 s':>9}{'最大 s':>10}  判断")
for tag in sorted(f[:-6] for f in os.listdir('.') if f.endswith('.jsonl')):
    recs = [json.loads(l) for l in open(f"{tag}.jsonl", encoding="utf-8")][1:]
    if not recs:
        continue
    lat = sorted(r["latency_s"] for r in recs)
    total, med, mx = sum(lat), lat[len(lat)//2], lat[-1]
    flag = "⚠️ 有长尾，需重跑" if mx > 0.2 * total else "✅ 正常"
    print(f"  {tag:<20}{total:>10.2f}{med:>9.3f}{mx:>10.3f}  {flag}")
PYEOF
echo
echo "JIT_CHECK_DONE"
