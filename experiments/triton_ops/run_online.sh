#!/usr/bin/env bash
# 入口：在线 softmax —— 验证 rescale 是必需的
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/../vlm_inference_benchmark/shared/paths.sh" 2>/dev/null || true

PY="${VLLM_PYTHON_BIN:-$HOME/venvs/vllm/bin/python}"

echo "[env] python: $PY"
echo "[env] GPU 基线: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo
exec "$PY" "$SCRIPT_DIR/online_softmax_toy.py" "$@"
