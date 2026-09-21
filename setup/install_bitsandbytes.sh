#!/usr/bin/env bash
# 安装 bitsandbytes（NF4 量化依赖）· --no-deps 保护 vLLM 依赖树
set -uo pipefail
PY="$HOME/venvs/vllm/bin/python"

echo "=== 安装前状态 ==="
"$PY" -m pip show bitsandbytes 2>/dev/null | head -3 || echo "(bitsandbytes 未安装)"
echo
echo "=== 安装 bitsandbytes（--no-deps：不动 numpy/torch/transformers）==="
"$PY" -m pip install --no-deps bitsandbytes 2>&1 | tail -12
echo
echo "=== 验证 bitsandbytes 可导入 ==="
"$PY" -c "import bitsandbytes as bnb; print('bnb', bnb.__version__)" 2>&1 | tail -6
echo
echo "=== 回归验证：vLLM 依赖树未被打坏 ==="
"$PY" -c "import vllm, transformers, torch, numpy; print('vllm', vllm.__version__, '| transformers', transformers.__version__, '| torch', torch.__version__, '| numpy', numpy.__version__)" 2>&1 | tail -6
echo "PIP_DONE"
