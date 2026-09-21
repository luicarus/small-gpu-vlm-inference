#!/usr/bin/env bash
# 探针：在 gmu 上限 0.80 的约束下，找出**能让 batch≥2 启动**的 vLLM 配置。
#
# 问题：AWQ 权重 2.74 GiB + offload=0 → gmu 0.80（预算 3277 MiB）时
#   b1: non_kv 2713 MiB → KV 池 +564 MiB ✅
#   b2: non_kv 3451 MiB → KV 池 −174 MiB ❌（死法 #2）
# 需要省出 ~200+ MiB。候选杠杆：
#   ① enforce_eager=True   —— 关掉 CUDA graph 捕获（batch=2 时捕获尺寸从 [1,2] 增到 [1,2,4]）
#   ② max_num_batched_tokens 1024→512 —— 压缩 profiling 那一次 forward 的激活
#   ③ max_pixels 262144→65536 —— 减少每张图的视觉 token，直接砍 ViT 激活
#
# 用单图冒烟（快，约 30s/次）逐个试，只关心"引擎能否起来 + KV 池多大"。
set -uo pipefail

# 路径锚点：向上找 small-gpu-vlm-inference，与脚本所在层级解耦（见 shared/paths.sh 说明）
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../shared" && pwd)/paths.sh"
bench_init "${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${VLLM_PYTHON_BIN:-$HOME/venvs/vllm/bin/python}"
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
    echo "🛑 [$tag] 基线脏（${used} MiB），终止探针 —— 请重启显卡驱动后重跑"
    exit 2
  fi

  echo
  echo "=============================================================="
  echo "  [$tag]  max_num_seqs=$seqs  额外参数: $*"
  echo "=============================================================="
  # 用 --max-num-seqs 传入；其余参数透传
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
    echo "  ✅ 启动成功   $kv   $concurrent"
  else
    local why
    why=$(grep -oE 'No available memory for the cache blocks|is less than desired GPU memory utilization|Available KV cache memory: -[0-9.]+ GiB' "$LOG/$tag.log" | tail -1)
    echo "  ❌ 启动失败   ${why:-（见日志 $LOG/$tag.log）}"
  fi
  sleep 6
}

echo "###### 基线（batch=1 配置，已知可行）######"
try_cfg "A_seqs1_base" 1
echo
echo "###### batch=2：逐个杠杆测试 ######"
try_cfg "B_seqs2_eager"  2 --enforce-eager
try_cfg "C_seqs2_mnbt512" 2 --max-num-batched-tokens 512
try_cfg "D_seqs2_px65536" 2 --max-pixels 65536
try_cfg "E_seqs2_eager_px" 2 --enforce-eager --max-pixels 65536

echo
echo "PROBE_DONE"