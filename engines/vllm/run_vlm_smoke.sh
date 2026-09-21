#!/bin/bash
set -euo pipefail

export HF_HUB_DISABLE_XET=1
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${VLLM_PYTHON_BIN:-$HOME/venvs/vllm/bin/python}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/vlm_smoke.py" \
  --image "$SCRIPT_DIR/../../assets/test_image.png" \
  --cpu-offload-gb "${VLM_CPU_OFFLOAD_GB:-1.0}" \
  --gpu-memory-utilization "${VLM_GPU_MEMORY_UTILIZATION:-0.70}" \
  --max-model-len "${VLM_MAX_MODEL_LEN:-1024}" \
  --max-num-seqs "${VLM_MAX_NUM_SEQS:-1}" \
  "$@"
