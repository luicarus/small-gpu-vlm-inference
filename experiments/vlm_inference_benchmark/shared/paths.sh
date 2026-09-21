#!/usr/bin/env bash
# vlm_inference_benchmark 共用 shell 工具
#
# 为什么需要它：所有实验脚本都要定位到 small-gpu-vlm-inference（放 results/）和 docs/（放图片）。
# 如果用固定层级的 `../..`，一旦脚本被移动（比如从 experiments/w5_xxx 挪到
# experiments/vlm_inference_benchmark/exp2_xxx），相对层级就变了，**所有落盘路径会静默出错**。
# 这里改成「向上找锚点」：认得出同时含 `engines/` 与 `results/` 的那一级就是 small-gpu-vlm-inference。

# 向上查找 small-gpu-vlm-inference 根目录。$1 = 起点目录（通常是脚本所在目录）
find_infra_root() {
  local d="${1:-$PWD}"
  d="$(cd -- "$d" && pwd)"
  while [ "$d" != "/" ]; do
    if [ -d "$d/engines" ] && [ -d "$d/results" ]; then
      echo "$d"
      return 0
    fi
    d="$(dirname "$d")"
  done
  echo "❌ 找不到 small-gpu-vlm-inference 根（应同时含 engines/ 与 results/）" >&2
  return 1
}

# 由脚本自身位置推出**属于它的**结果目录。
#
# 约定：results/ 与 experiments/ 同名同构。
#   experiments/vlm_inference_benchmark/vllm/exp1_memory_boundary/run.sh
#   → results/vlm_inference_benchmark/vllm/exp1_memory_boundary/
# 为什么这么设计：实验会随研究推进被重新分组（例如把 vLLM 的实验收进 vllm/），
# 手写结果路径一重组就要全文改，极易漏；镜像规则让分组变化自动生效。
bench_results_dir() {
  local caller="${1:-${BASH_SOURCE[1]}}"
  local d
  d="$(cd -- "$(dirname -- "$caller")" && pwd)"
  echo "$RESULTS_ROOT${d#"$BENCH_ROOT"}"
}

# 设置共用变量：INFRA_ROOT / BENCH_ROOT / RESULTS_ROOT
# 用法：source ".../shared/paths.sh"; bench_init "${BASH_SOURCE[0]}"
bench_init() {
  local caller="${1:-${BASH_SOURCE[1]}}"
  local script_dir
  script_dir="$(cd -- "$(dirname -- "$caller")" && pwd)"
  INFRA_ROOT="$(find_infra_root "$script_dir")"
  BENCH_ROOT="$INFRA_ROOT/experiments/vlm_inference_benchmark"
  RESULTS_ROOT="$INFRA_ROOT/results/vlm_inference_benchmark"
  export INFRA_ROOT BENCH_ROOT RESULTS_ROOT
}

# 基线守卫：整卡占用超过阈值就中止（WDDM 显存泄漏 / 上一轮残留的兜底）
#
# 阈值 1000 MiB 的依据：干净基线约 8~700 MiB（桌面波动），
# 而 vLLM 初始化失败留下的残留通常是数百 MiB 到 3000+ MiB ——
# **实测踩过一次 687 MiB 的残留**：它低于 1000 所以被放行，
# 但足以让下一个配置的 profiling 量到偏大的 non_kv（2610→3080 MiB）而被判"假失败"。
# 所以阈值收紧到 800，并且**建议配合 exp1 的 CUDA 侧门槛一起用**（两套口径要同时干净）。
require_clean_gpu() {
  local max_mib="${1:-800}" used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "$used" -gt "$max_mib" ]; then
    echo "🛑 基线守卫：整卡占用 ${used} MiB > 阈值 ${max_mib} MiB —— 终止。" >&2
    echo "   多为上一次 vLLM 初始化失败留下的显存残留（进程已退出但驱动未回收）。" >&2
    echo "   处理：按 Win+Ctrl+Shift+B 重启显卡驱动（无效则重启 Windows），再重跑。" >&2
    return 1
  fi
  # 注意：NVML 干净**不代表** CUDA 侧也干净（反之亦然）——
  # 两套口径在驱动回收滞后时会不一致。需要严格判定时用 exp1 的 wait_for_memory_release。
  echo "[guard] 整卡 ${used} MiB ✅"
}

# Python 解释器（vLLM venv；torch 只装在这里，系统 python3 没有）
bench_python() {
  echo "${VLLM_PYTHON_BIN:-$HOME/venvs/vllm/bin/python}"
}