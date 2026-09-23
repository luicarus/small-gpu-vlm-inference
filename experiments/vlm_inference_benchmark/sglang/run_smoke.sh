#!/usr/bin/env bash
# W11 冒烟入口：SGLang 在 4GB 上跑 Qwen2-VL-2B-AWQ
#
# ⚠️ 两个必须的环境设置（都是实测踩出来的）：
#
# 1. **独立 venv**：SGLang 要的 transformers 版本与 vLLM 不同（5.12.1 vs 5.16.1），
#    混装会**静默**破坏已跑通的三组实验环境。
#
# 2. **CUDA_HOME 指向 venv 内的 CUDA 13 工具链**：
#    flashinfer 会 JIT 编译 attention kernel，命令行里带 `--compress-mode=size`
#    （CUDA 12.1+ 才有的选项）。而本机系统 nvcc 是 **CUDA 12.0**，直接报
#    `nvcc fatal : Unknown option '--compress-mode=size'` → ninja build 全失败
#    → CUDA graph 捕获失败 → 引擎起不来。
#    flashinfer 的选择顺序是 `CUDA_HOME`/`CUDA_PATH` 优先于 `which nvcc`，
#    所以指到 venv 自带的 cu13 即可（实测该 nvcc 为 13.4，支持该选项）。
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 路径锚点：向上找仓库根（复用主实验的 shared）
source "$SCRIPT_DIR/../shared/paths.sh"
bench_init "${BASH_SOURCE[0]}"

PY="${SGLANG_PYTHON_BIN:-$HOME/venvs/sglang/bin/python}"

# 让 flashinfer 用 venv 里的新 nvcc，而不是系统的 CUDA 12.0
VENV_CUDA="$(dirname "$PY")/../lib/python3.12/site-packages/nvidia/cu13"
if [ -d "$VENV_CUDA" ]; then
  export CUDA_HOME="$VENV_CUDA"
  export CUDA_PATH="$VENV_CUDA"
  # 同时把它的 bin 放到 PATH 前面，覆盖 `which nvcc` 的结果
  export PATH="$VENV_CUDA/bin:$PATH"
fi

echo "[env] sglang python : $PY"
echo "[env] CUDA_HOME     : ${CUDA_HOME:-（未设置）}"
echo "[env] which nvcc    : $(which nvcc)"
"$PY" -c "import sglang; print('[env] sglang', sglang.__version__)"

require_clean_gpu || exit 2

exec "$PY" "$SCRIPT_DIR/smoke.py" "$@"