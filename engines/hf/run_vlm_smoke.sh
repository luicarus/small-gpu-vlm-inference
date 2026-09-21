#!/bin/bash
# HF NF4 冒烟入口（与 engines/vllm/run_vlm_smoke.sh 对称，参数名保持一致便于对比）
set -euo pipefail

export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
# HF 侧没有 vLLM 的 cpu_offload_gb，可用内存上限由 WSL 虚拟机决定（本机 7.6 GiB）
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${HF_PYTHON_BIN:-$HOME/venvs/vllm/bin/python}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/vlm_smoke.py" \
  --image "$SCRIPT_DIR/../../assets/test_image.png" \
  --quant "${HF_QUANT:-nf4}" \
  --max-new-tokens "${HF_MAX_NEW_TOKENS:-64}" \
  --max-pixels "${HF_MAX_PIXELS:-50176}" \
  "$@"
