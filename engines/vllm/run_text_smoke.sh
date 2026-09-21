#!/bin/bash
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd ~
~/venvs/vllm/bin/python "$SCRIPT_DIR/text_smoke.py" 2>&1
echo "EXIT=$?"
