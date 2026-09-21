#!/usr/bin/env bash
# 共享前缀长度对加速比的影响（帧数扫描）
#
# 目的：验证「收益随前缀长度增长」这个外推结论。
# 前一阶段（16 帧 / 2592 token 前缀）实测 TTFT 加速约 2×，
# 拆解出「固定开销 C≈0.435s + 每 token 0.169ms」，据此外推前缀越长加速越大。
#
# 帧数档位怎么定的（依据 probe_frames.sh 实测）：
#   每帧约 162 token；36 帧时 KV 池告警「max_model_len 上限 7440」
#   （死法 #3：KV 池装不下一条 max_model_len 的序列）
#   → 取 8 / 16 / 32 / 40 帧，prefix ≈ 1296 / 2592 / 5184 / 6480 token
#   → max_model_len 相应设为 2048 / 4096 / 6144 / 7168
#
# 每组都要跑「缓存关 + 缓存开」两份，才能算加速比。
set -uo pipefail

# 路径锚点：向上找 small-gpu-vlm-inference，与脚本所在层级解耦（见 shared/paths.sh 说明）
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../shared" && pwd)/paths.sh"
bench_init "${BASH_SOURCE[0]}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${VLLM_PYTHON_BIN:-$HOME/venvs/vllm/bin/python}"
RESULTS="$(bench_results_dir "${BASH_SOURCE[0]}")"
LOGS="$RESULTS/logs"
mkdir -p "$LOGS"

export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
export VLLM_WSL2_ENABLE_PIN_MEMORY=1 HF_HUB_DISABLE_XET=1

MAX_BASE_MIB="${MAX_BASE_MIB:-1000}"
N="${N:-10}"
FRAME_CONFIGS="${FRAME_CONFIGS:-8:2048 16:4096 32:6144 40:7168}"

require_clean_gpu() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "$used" -gt "$MAX_BASE_MIB" ]; then
    echo "🛑 基线守卫：${used} MiB > ${MAX_BASE_MIB} —— 终止（多为上次失败的 WDDM 泄漏）"
    echo "   按 Win+Ctrl+Shift+B 重启显卡驱动后重跑。已完成的结果仍有效。"
    exit 2
  fi
  echo "[guard] 整卡 ${used} MiB ✅"
}

run_one() {
  local cache="$1" frames="$2" mml="$3"
  local tag="frames${frames}_cache${cache}"
  echo
  echo "=============================================================="
  echo "  [$tag]  frames=$frames  max_model_len=$mml  caching=$cache  N=$N"
  echo "=============================================================="
  require_clean_gpu

  local extra=()
  [ "$cache" = "off" ] && extra=(--no-prefix-caching)

  "$PY" "$SCRIPT_DIR/run_cache.py" \
    --num-questions "$N" --frames "$frames" --max-model-len "$mml" "${extra[@]}" \
    --out "$RESULTS/${tag}.jsonl" 2>&1 | tee "$LOGS/${tag}.log" \
    | grep -E '冷启动|后续（命中）|平均命中率|参考：vLLM|整轮墙钟|BENCH_RUN_OK|ValueError|No available|Error' \
    || echo "  ⚠️ 该组失败（见 $LOGS/${tag}.log）"

  sleep 8
  echo "跑完整卡占用：$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}

echo "##############################"
echo "# 帧数扫描实验 $(date '+%F %T')"
echo "# 档位：$FRAME_CONFIGS   （frames:max_model_len）"
echo "##############################"

for cfg in $FRAME_CONFIGS; do
  frames="${cfg%%:*}"
  mml="${cfg##*:}"
  run_one off "$frames" "$mml"
  run_one on  "$frames" "$mml"
done

echo
echo "##############################"
echo "# 完成 $(date '+%F %T')"
echo "##############################"
ls -la "$RESULTS"/frames*.jsonl 2>/dev/null
echo "FRAMES_DONE"