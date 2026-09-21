#!/usr/bin/env bash
# S1 可行性冒烟：确认 Qwen2-VL-2B 上两条路线都能跑通，再决定是否建正式实验。
#
# 两个待验证的关键点（都是 3B 上失败过的地方）：
#   1. HF + NF4 且 **ViT 保持 bf16** —— 3B 上撞过 transformers/bnb 混精度缺陷（AssertionError）
#      + 显存不足（5.4 GiB）。2B 预算约 2.55 GiB，理论可行，但缺陷是否复现未知。
#   2. vLLM + AWQ 且 **cpu_offload_gb=0** —— 3B 上 offload=0 必死于死法 #2（权重 3.17 GiB 装不下）。
#      2B 权重 2.74 GiB，理论上 gmu 0.80 能挤下，但要实测确认。
#
# 注意：不改 vlm_smoke.py 的 DEFAULT_MODEL（否则其它脚本会静默换模型），一律用环境变量显式指定。
set -uo pipefail

# 路径锚点：向上找 small-gpu-vlm-inference，与脚本所在层级解耦（见 shared/paths.sh 说明）
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../shared" && pwd)/paths.sh"
bench_init "${BASH_SOURCE[0]}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENGINES="$INFRA_ROOT/engines"
HF_MODEL="$HOME/models/Qwen2-VL-2B-Instruct"
AWQ_MODEL="$HOME/models/Qwen2-VL-2B-Instruct-AWQ"

# 【基线守卫】2026-09-18 踩坑：一次冒烟在基线 4069 MiB（只剩 27 MiB）的污染环境下开跑，
# 失败结果（AssertionError）事后无法判定是"真 bug"还是"被掩盖的 OOM"，整轮作废。
# 桌面占用正常范围是 134~800 MiB；超过阈值一律拒绝运行。
MAX_BASE_MIB="${MAX_BASE_MIB:-1000}"

check_gpu_clean() {
  local phase="$1"
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  echo "[guard:$phase] 整卡占用 ${used} MiB（阈值 ${MAX_BASE_MIB}）"
  if [ "$used" -gt "$MAX_BASE_MIB" ]; then
    echo "❌ 基线过高，拒绝运行。"
    echo "   历史经验：这种情况多半是 WDDM 显存泄漏（进程已死但驱动未回收）。"
    echo "   处理：按 Win+Ctrl+Shift+B 重启显卡驱动；仍无效则重启 Windows。"
    echo "   若确认是有意为之，可设 MAX_BASE_MIB=<更大值> 跳过守卫。"
    return 1
  fi
  return 0
}

for m in "$HF_MODEL" "$AWQ_MODEL"; do
  [ -d "$m" ] || { echo "❌ 模型缺失: $m"; exit 1; }
done

echo "############ GPU 状态 ############"
bash "$INFRA_ROOT/setup/gpu_check.sh"
check_gpu_clean "启动前" || exit 2

echo
echo "############ 1/2  HF: NF4 + ViT 保持 bf16（对齐 AWQ 量化范围）############"
HF_MODEL="$HF_MODEL" bash "$ENGINES/hf/run_vlm_smoke.sh" --keep-vision-bf16 2>&1 \
  | grep -vE 'Loading weights|it/s\]' | tail -32

echo
sleep 6
check_gpu_clean "HF 跑完" || echo "⚠️ HF 跑完显存未回落，vLLM 结果可能受影响"

echo "############ 2/2  vLLM: AWQ + cpu_offload_gb=0（去掉 PCIe 税）############"
VLM_MODEL="$AWQ_MODEL" \
VLM_CPU_OFFLOAD_GB=0 \
VLM_GPU_MEMORY_UTILIZATION=0.80 \
bash "$ENGINES/vllm/run_vlm_smoke.sh" 2>&1 | tail -32

echo
echo "############ 冒烟结束，整卡占用 ############"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "S1_SMOKE_DONE"