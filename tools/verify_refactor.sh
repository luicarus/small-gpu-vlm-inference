#!/usr/bin/env bash
# 结构回归自检：确认脚本仍能找到彼此、结果仍完整、引用没有断链。
#
# 全部检查都不占 GPU，约 1 分钟。改动目录结构或脚本路径后跑一次。
set -uo pipefail

# 以仓库根目录为基准（本脚本在 tools/ 下）
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
PY="${VLLM_PYTHON_BIN:-$HOME/venvs/vllm/bin/python}"
BENCH="experiments/vlm_inference_benchmark"
VLLM="$BENCH/vllm"
RB="results/vlm_inference_benchmark/vllm"

echo "===== 1. 路径锚点 + 结果目录镜像规则 ====="
"$PY" - <<PYEOF
import sys
sys.path.insert(0, "$REPO/$BENCH")
from pathlib import Path
from shared.common import INFRA_ROOT, BENCH_ROOT, RESULTS_ROOT, results_dir_of, MODEL_AWQ, DEFAULT_IMAGE
assert (INFRA_ROOT / "engines").is_dir(), "路径锚点找错了"
assert BENCH_ROOT.name == "vlm_inference_benchmark"
assert RESULTS_ROOT.name == "vlm_inference_benchmark"
assert DEFAULT_IMAGE.exists(), "测试图片缺失"
# 四个实验的结果目录都要能镜像出来
for exp in ("exp1_memory_boundary", "exp2_engine_comparison",
            "exp3_prefix_caching", "preliminary_3b_memory_sweep"):
    d = results_dir_of(BENCH_ROOT / "vllm" / exp / "x.py")
    assert d.is_dir(), f"镜像目录不存在: {d}"
print("  ✅ 锚点正确 · 图片在位 · 4 个结果目录镜像成功")
print(f"  （模型路径 {MODEL_AWQ} —— 不存在也能跑分析，只影响重跑实验）")
PYEOF

echo
echo "===== 2. exp1 显存边界：dry-run + 汇总 ====="
"$PY" "$VLLM/exp1_memory_boundary/sweep.py" --dry-run 2>&1 | head -3
"$PY" "$VLLM/exp1_memory_boundary/summarize.py" 2>&1 | grep -E "来源|OK " | head -2

echo
echo "===== 3. exp2 双栈对比：四指标分析 ====="
"$PY" "$VLLM/exp2_engine_comparison/compare.py" 2>&1 | grep -E "^\| (hf|vllm)" | head -4

echo
echo "===== 4. exp3 workload 生成器 ====="
"$PY" "$VLLM/exp3_prefix_caching/make_workload.py" --out /tmp/bench_verify.json 2>&1 | head -3

echo
echo "===== 5. 结果文件完整性 ====="
miss=0
for b in 1 2 4 8; do
  for e in hf_nf4 vllm_awq; do
    [ -s "$RB/exp2_engine_comparison/${e}_b${b}.jsonl" ] || { echo "  ❌ 缺失 ${e}_b${b}"; miss=1; }
  done
done
for f in 8 16 32 40; do
  for c in off on; do
    [ -s "$RB/exp3_prefix_caching/frames${f}_cache${c}.jsonl" ] || { echo "  ❌ 缺失 frames${f}_${c}"; miss=1; }
  done
done
ls "$RB/exp1_memory_boundary"/exp1_sweep_*.jsonl >/dev/null 2>&1 || { echo "  ❌ 缺 exp1 扫描结果"; miss=1; }
ls results/bnb_align/*.jsonl >/dev/null 2>&1 || { echo "  ❌ 缺 bnb 微基准结果"; miss=1; }
[ $miss -eq 0 ] && echo "  ✅ 齐全（exp1 + exp2 ×8 + exp3 ×8 + bnb）"

echo
echo "===== 6. 报告数字与结果文件一致性 ====="
if [ -f REPORT.md ]; then
  "$PY" - "$REPO/$RB" "$REPO/REPORT.md" <<'PYEOF'
import json, sys, os, statistics as st
RB, DOC = sys.argv[1], sys.argv[2]
doc = open(DOC, encoding="utf-8").read()
def load(p):
    recs = [json.loads(l) for l in open(p, encoding="utf-8")]
    return recs[0]["_meta"], recs[1:]
miss = []
for b in ("1","2","4","8"):
    for e in ("hf_nf4","vllm_awq"):
        p = f"{RB}/exp2_engine_comparison/{e}_b{b}.jsonl"
        if not os.path.exists(p): continue
        m, _ = load(p)
        if str(m["peak_mib"]) not in doc: miss.append(f"exp2 {e}_b{b} 峰值 {m['peak_mib']}")
for fr in (8,16,32,40):
    for c in ("off","on"):
        p = f"{RB}/exp3_prefix_caching/frames{fr}_cache{c}.jsonl"
        if not os.path.exists(p): continue
        _, recs = load(p)
        med = st.median([r["wall_s"] for r in recs[1:]])
        if f"{med:.3f}" not in doc: miss.append(f"exp3 frames{fr}_{c} 中位 {med:.3f}")
print("  ✅ 报告关键数字与结果文件一致" if not miss else
      "  ⚠️ 报告里找不到这些实测值：\n" + "\n".join(f"    {x}" for x in miss))
PYEOF
else
  echo "  (无 REPORT.md)"
fi

echo
echo "===== 7. 语法编译检查 ====="
"$PY" -m compileall -q engines experiments tools && echo "  全部通过" || echo "  有语法错误"
find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo
echo "===== 8. 残留引用扫描 ====="
STALE='infra-scripts|vlm_4gb_benchmark|/home/luxing|AGENTS\.md|docs/周记录|docs/blog|docs/笔记'
hits=$(grep -rnE "$STALE" --include="*.sh" --include="*.py" --include="*.md" \
        engines experiments tools setup assets README.md REPORT.md 2>/dev/null \
        | grep -vE "_archive|verify_refactor" || true)
if [ -n "$hits" ]; then
  echo "  ❌ 发现不该出现的引用："
  echo "$hits" | sed 's/^/    /'
else
  echo "  ✅ 无残留引用"
fi

echo
echo "===== 9. 固定层级引用扫描（只查会搬动的 experiments/）====="
hier=$(grep -rnE '(parents\[[0-9]+\]|SCRIPT_DIR/\.\./\.\.)[^)]*(results|docs)' \
        --include="*.py" --include="*.sh" experiments 2>/dev/null \
        | grep -vE 'sys\.path\.insert' || true)
[ -n "$hier" ] && { echo "  ⚠️ 仍有固定层级引用："; echo "$hier" | sed 's/^/    /'; } \
               || echo "  ✅ experiments/ 下全部使用路径锚点/镜像规则"

echo
echo "REGRESSION_CHECK_DONE"
