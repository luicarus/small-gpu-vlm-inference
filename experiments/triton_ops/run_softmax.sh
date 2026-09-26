#!/usr/bin/env bash
# 入口：Fused Softmax 教程（融合省 IO + 与 FA 的差距对照）
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/../vlm_inference_benchmark/shared/paths.sh" 2>/dev/null || true

PY="${VLLM_PYTHON_BIN:-$HOME/venvs/vllm/bin/python}"

echo "[env] python: $PY"
echo "[env] GPU 基线: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo
exec "$PY" "$SCRIPT_DIR/softmax_tutorial.py" "$@"
