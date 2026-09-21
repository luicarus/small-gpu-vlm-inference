#!/usr/bin/env bash
# 安装 accelerate（device_map="auto" 兜底路径依赖）
set -uo pipefail
PY="$HOME/venvs/vllm/bin/python"
"$PY" -m pip install --no-deps accelerate 2>&1 | tail -4
"$PY" - <<'EOF'
try:
    import accelerate
    print("accelerate", accelerate.__version__)
except Exception as exc:
    print("ACCELERATE_IMPORT_FAIL:", exc)
EOF
echo "=== 回归验证 vLLM 依赖树 ==="
"$PY" -c "import vllm, transformers, torch, bitsandbytes; print('vllm', vllm.__version__, '| transformers', transformers.__version__, '| torch', torch.__version__, '| bnb', bitsandbytes.__version__)" 2>&1 | tail -3
echo "=== 已下载的模型 ==="
du -sh "$HOME/models"/*/ 2>/dev/null || echo "(尚无模型)"
echo "ACCEL_DONE"
