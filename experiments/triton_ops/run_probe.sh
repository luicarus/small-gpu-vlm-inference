#!/usr/bin/env bash
# 前置调查：Triton 现状
#
# Triton 其实已经作为 vLLM / SGLang 的依赖装上了，所以"安装"这一步的真实问题不是
# "能不能装"，而是：
#   ① 两个 venv 里的 triton 版本分别是什么；
#   ② 能否在 sm86（RTX 3050 Ti）上真正编译并跑通一个 kernel；
#   ③ 是否需要独立 venv（避免动到已跑通的实验环境）。
set -uo pipefail

echo "===== 1. 两个 venv 里的 triton 版本 ====="
echo "  --- vllm venv ---"
"$HOME/venvs/vllm/bin/python" -m pip list 2>/dev/null | grep -iE "^triton" || echo "    (无)"
echo "  --- sglang venv ---"
"$HOME/venvs/sglang/bin/python" -m pip list 2>/dev/null | grep -iE "^triton|tokenspeed-triton" || echo "    (无)"

echo
echo "===== 2. triton 能否 import（vllm venv）====="
"$HOME/venvs/vllm/bin/python" - <<'PYEOF'
try:
    import triton
    import triton.language as tl
    print(f"  ✅ triton {triton.__version__}")
    print(f"     文件: {triton.__file__}")
except Exception as e:
    print(f"  ❌ {type(e).__name__}: {str(e)[:150]}")
PYEOF

echo
echo "===== 3. GPU 与算力 ====="
"$HOME/venvs/vllm/bin/python" - <<'PYEOF'
import torch
print(f"  device    : {torch.cuda.get_device_name(0)}")
cap = torch.cuda.get_device_capability()
print(f"  capability: sm{cap[0]}{cap[1]}")
print(f"  torch     : {torch.__version__}")
PYEOF

echo
echo "===== 4. nvcc 情况（Triton 通常不需要它，但需要 driver）====="
nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed 's/^/  driver: /'
nvcc --version 2>/dev/null | tail -1 | sed 's/^/  nvcc:   /' || echo "  nvcc:   (系统无)"

echo
echo "===== 5. 编译缓存目录 ====="
ls -d ~/.triton/cache 2>/dev/null && du -sh ~/.triton/cache 2>/dev/null || echo "  (尚未创建，首次编译时生成)"
echo
echo "TRITON_PROBE_DONE"