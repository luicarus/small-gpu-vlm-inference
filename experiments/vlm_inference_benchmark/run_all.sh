#!/usr/bin/env bash
# 一键跑完三组实验（严格串行 —— 只有一块 4GB 卡）
#
# 顺序有讲究：exp1 先跑，因为它给出的 non_kv / KV 池 / KV-per-token
# 是解释 exp2、exp3 结果的量化基础；而且它会顺带确认基线是否干净。
set -uo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/shared" && pwd)/paths.sh"
bench_init "${BASH_SOURCE[0]}"

require_clean_gpu || {
  echo "基线不干净，请先重启显卡驱动（Win+Ctrl+Shift+B）" >&2
  exit 2
}

echo "##############################"
echo "# vlm_inference_benchmark 全量实验  $(date '+%F %T')"
echo "# 模型：Qwen2-VL-2B-Instruct-AWQ（offload=0）"
echo "# 预计总时长约 45 分钟"
echo "##############################"

run_one() {
  local name="$1" script="$2"
  echo
  echo "=============================================================="
  echo "  ▶ $name"
  echo "=============================================================="
  if ! require_clean_gpu; then
    echo "🛑 $name 之前基线已脏，终止全量流程。" >&2
    echo "   重启显卡驱动后可从单独入口续跑该实验。" >&2
    exit 2
  fi
  bash "$script" || echo "⚠️ $name 退出码非 0（检查上面的输出）"
}

run_one "exp1 显存边界扫描"   "$BENCH_ROOT/vllm/exp1_memory_boundary/run.sh"
run_one "exp2 双栈对比"       "$BENCH_ROOT/vllm/exp2_engine_comparison/run.sh"
run_one "exp3 前缀缓存实验"   "$BENCH_ROOT/vllm/exp3_prefix_caching/run_frames.sh"

echo
echo "##############################"
echo "# 全部完成 $(date '+%F %T')"
echo "##############################"
echo "--- 结果 ---"
find "$RESULTS_ROOT" -name "*.jsonl" -printf "  %p\n" 2>/dev/null | sort
echo
echo "--- 汇总 ---"
"$(bench_python)" "$BENCH_ROOT/vllm/exp1_memory_boundary/summarize.py" 2>/dev/null | head -8
echo "ALL_DONE"