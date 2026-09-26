#!/usr/bin/env bash
# 入口：FlashAttention 骨架 + 峰值显存扫描（本主题的核心实验）
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/../vlm_inference_benchmark/shared/paths.sh" 2>/dev/null || true

PY="${VLLM_PYTHON_BIN:-$HOME/venvs/vllm/bin/python}"

echo "[env] python: $PY"
echo "[env] GPU 基线: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo
exec "$PY" "$SCRIPT_DIR/flash_attention_toy.py" "$@"
