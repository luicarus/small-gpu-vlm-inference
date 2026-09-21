#!/usr/bin/env bash
# bnb 内维对齐快慢路径微基准入口（不加载模型，显存占用极低）
#
# 这个实验与模型无关（纯微基准），所以不进 vlm_inference_benchmark，独立成一个目录。
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 路径锚点：向上找 small-gpu-vlm-inference，与脚本所在层级解耦。
# 借用 benchmark 的 shared/paths.sh（同一份锚点逻辑，不重复实现）。
source "$SCRIPT_DIR/../vlm_inference_benchmark/shared/paths.sh"
bench_init "${BASH_SOURCE[0]}"

PY="$(bench_python)"
RESULTS="$INFRA_ROOT/results/bnb_align"

mkdir -p "$RESULTS"
exec "$PY" "$SCRIPT_DIR/bench_align.py" --out "$RESULTS/bench_align.jsonl" "$@"