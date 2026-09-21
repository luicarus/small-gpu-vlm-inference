#!/usr/bin/env bash
# 冒烟入口：验证伪视频共享前缀这条路
set -uo pipefail

# 路径锚点：向上找 small-gpu-vlm-inference，与脚本所在层级解耦（见 shared/paths.sh 说明）
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../shared" && pwd)/paths.sh"
bench_init "${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${VLLM_PYTHON_BIN:-$HOME/venvs/vllm/bin/python}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=1
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export HF_HUB_DISABLE_XET=1

exec "$PY" "$SCRIPT_DIR/smoke_video.py" "$@"