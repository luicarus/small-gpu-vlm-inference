#!/usr/bin/env bash
# prefix caching 开/关 × 问题数 N 的对比
#
# 设计（为什么这么跑）：
# 1. **串行**：一块 4GB 卡只允许一个推理进程。
# 2. **先热身再取数**：Triton 运行时会为没见过的 shape 编译 kernel（实测），
#    首次运行会有秒级长尾。本脚本先跑一轮小 N 当 warmup 丢弃，再取正式数据。
# 3. **配置矩阵**：prefix caching {on, off} × N {5, 20, 50}
#    —— 验证收益是否随 N 增长，以及是否符合 N(P+Q)/(P+NQ)。
set -uo pipefail

# 路径锚点：向上找 small-gpu-vlm-inference，与脚本所在层级解耦（见 shared/paths.sh 说明）
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../shared" && pwd)/paths.sh"
bench_init "${BASH_SOURCE[0]}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${VLLM_PYTHON_BIN:-$HOME/venvs/vllm/bin/python}"
RESULTS="$(bench_results_dir "${BASH_SOURCE[0]}")"
LOGS="$RESULTS/logs"
mkdir -p "$LOGS"

export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=1
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export HF_HUB_DISABLE_XET=1

MAX_BASE_MIB="${MAX_BASE_MIB:-1000}"
NS="${NS:-5 20 50}"

require_clean_gpu() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "$used" -gt "$MAX_BASE_MIB" ]; then
    echo "🛑 基线守卫：整卡占用 ${used} MiB > ${MAX_BASE_MIB} —— 终止。"
    echo "   多为 vLLM 初始化失败留下的 WDDM 显存泄漏；按 Win+Ctrl+Shift+B 重启显卡驱动后重跑。"
    exit 2
  fi
  echo "[guard] 整卡 ${used} MiB ✅"
}

run_one() {
  local cache="$1" n="$2"
  local tag="cache${cache}_n${n}"
  echo
  echo "=============================================================="
  echo "  [$tag]  prefix_caching=$cache  N=$n"
  echo "=============================================================="
  require_clean_gpu

  local extra=()
  [ "$cache" = "off" ] && extra=(--no-prefix-caching)

  "$PY" "$SCRIPT_DIR/run_cache.py" \
    --num-questions "$n" "${extra[@]}" \
    --out "$RESULTS/${tag}.jsonl" 2>&1 | tee "$LOGS/${tag}.log" | tail -18
  sleep 8
  echo "跑完整卡占用：$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}

echo "##############################"
echo "# prefix caching 实验开始 $(date '+%F %T')"
echo "# workload：同一张图重复成伪视频（16 帧）作共享前缀 + N 个不同问题"
echo "# 矩阵：prefix caching {on, off} × N {$NS}"
echo "##############################"

echo
echo "########## 热身（丢弃）：让 Triton 编译好这个 shape ############"
run_one on 2 >/dev/null 2>&1 || true
echo "热身完成"

for n in $NS; do
  run_one off "$n"
  run_one on  "$n"
done

echo
echo "##############################"
echo "# 实验完成 $(date '+%F %T')"
echo "##############################"
ls -la "$RESULTS"/*.jsonl 2>/dev/null
echo "BENCH_RUN_ALL_DONE"