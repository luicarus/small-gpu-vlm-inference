#!/usr/bin/env bash
# 探针 v2：确定 batch=4 的可行配置，并选出跨 batch 统一使用的参数。
# 探针 v1 已证明：max_num_seqs=2 时把 max_num_batched_tokens 从 1024 压到 512，
# KV 池反而更大（0.68 vs 失败），因为 profiling 那一次 forward 的激活被压缩了。
# 本轮专门试 batch=4（max_num_seqs=4）。
set -uo pipefail

# 路径锚点：向上找 small-gpu-vlm-inference，与脚本所在层级解耦（见 shared/paths.sh 说明）
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../shared" && pwd)/paths.sh"
bench_init "${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENGINES="$INFRA_ROOT/engines"
LOG="$(bench_results_dir "${BASH_SOURCE[0]}")/probe_logs"
mkdir -p "$LOG"

export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=1
export VLLM_WSL2_ENABLE_PIN_MEMORY=1

try_cfg() {
  local tag="$1" seqs="$2"; shift 2
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "$used" -gt 1000 ]; then
    echo "🛑 [$tag] 基线脏（${used} MiB）—— 终止探针，请重启显卡驱动后重跑"
    exit 2
  fi
  echo
  echo "=============================================================="
  echo "  [$tag]  max_num_seqs=$seqs  参数: $*"
  echo "=============================================================="
  VLM_MODEL="$HOME/models/Qwen2-VL-2B-Instruct-AWQ" \
  VLM_CPU_OFFLOAD_GB=0 \
  VLM_GPU_MEMORY_UTILIZATION=0.80 \
  VLM_MAX_NUM_SEQS="$seqs" \
    bash "$ENGINES/vllm/run_vlm_smoke.sh" "$@" >"$LOG/$tag.log" 2>&1
  local rc=$?

  if [ $rc -eq 0 ] && grep -q VLM_SMOKE_TEST_OK "$LOG/$tag.log"; then
    local kv concurrent
    kv=$(grep -oE 'Available KV cache memory: [0-9.]+ GiB' "$LOG/$tag.log" | tail -1)
    concurrent=$(grep -oE 'Maximum concurrency for 1,024 tokens per request: [0-9.]+x' "$LOG/$tag.log" | tail -1)
    echo "  ✅ 成功   $kv   $concurrent"
  else
    local why
    why=$(grep -oE 'No available memory for the cache blocks|is less than desired|Available KV cache memory: -[0-9.]+ GiB' "$LOG/$tag.log" | tail -1)
    echo "  ❌ 失败   ${why:-（见 $LOG/$tag.log）}"
  fi
  sleep 6
}

echo "###### batch=4 与 batch=8：找上限 ######"
try_cfg "F_seqs4_mnbt512" 4 --max-num-batched-tokens 512
try_cfg "G_seqs4_mnbt256" 4 --max-num-batched-tokens 256
try_cfg "H_seqs8_mnbt512" 8 --max-num-batched-tokens 512
echo
echo "###### 统一配置的 batch=1 复核（mnbt=512，全曲线一致）######"
try_cfg "I_seqs1_mnbt512" 1 --max-num-batched-tokens 512

echo
echo "PROBE2_DONE"