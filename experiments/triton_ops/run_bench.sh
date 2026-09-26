#!/usr/bin/env bash
# 入口：vector add 的 N 扫描（验证小负载差距来自启动开销）
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 复用主实验的路径锚点（保持项目内一致的解释器解析）
source "$SCRIPT_DIR/../vlm_inference_benchmark/shared/paths.sh" 2>/dev/null || true

PY="${VLLM_PYTHON_BIN:-$HOME/venvs/vllm/bin/python}"

echo "[env] python: $PY"
echo "[env] GPU 基线: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo
exec "$PY" "$SCRIPT_DIR/bench_add_sweep.py" "$@"
