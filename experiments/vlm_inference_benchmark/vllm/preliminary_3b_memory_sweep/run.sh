#!/usr/bin/env bash
# 前置实验入口：3B 模型上的显存边界扫描
#
# ⚠️ 这组实验已被 2B 版（../exp1_memory_boundary/）取代，模型也已删除。
#    保留的原因与结论见 ./README.md —— 它记录的是"发现混杂变量并重做"的过程。
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 路径锚点：向上找 small-gpu-vlm-inference（复用 benchmark 的 shared 实现，不重复一份）
source "$SCRIPT_DIR/../../shared/paths.sh"
bench_init "${BASH_SOURCE[0]}"

PY="$(bench_python)"

# 与 run_vlm_smoke.sh 保持一致的环境：离线加载 + WSL2 锁页内存 + 关掉 FlashInfer 采样器
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0

# 模型不在就直接说清楚，别让人对着报错猜
if [ ! -d "$HOME/models/Qwen2.5-VL-3B-Instruct-AWQ" ]; then
  echo "❌ 3B 模型不存在：$HOME/models/Qwen2.5-VL-3B-Instruct-AWQ"
  echo
  echo "   这组前置实验的模型已删除（释放 10.3 GiB），且它已被 2B 版取代："
  echo "     → 请跑 ../exp1_memory_boundary/run.sh"
  echo "   原因：3B 权重 3.17 GiB 装不下 4GB 卡（CUDA 可用仅 3.21 GiB），"
  echo "         vLLM 被迫 offload → 解码走 PCIe → 性能差距无法归因到引擎。"
  echo "   （本目录的历史数据汇总仍可跑：python summarize.py）"
  exit 3
fi

require_clean_gpu || exit 2

# 透传全部参数：--only / --timeout / --dry-run
exec "$PY" "$SCRIPT_DIR/sweep.py" "$@"