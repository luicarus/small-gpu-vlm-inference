#!/usr/bin/env bash
# Triton 可用性验证入口
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 复用主实验的路径锚点（虽然本步不落盘结果，但保持项目内一致的解释器解析）
source "$SCRIPT_DIR/../vlm_inference_benchmark/shared/paths.sh" 2>/dev/null || true

PY="${VLLM_PYTHON_BIN:-$HOME/venvs/vllm/bin/python}"

echo "[env] python: $PY"
echo "[env] GPU 基线: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo
exec "$PY" "$SCRIPT_DIR/check_triton.py" "$@"
