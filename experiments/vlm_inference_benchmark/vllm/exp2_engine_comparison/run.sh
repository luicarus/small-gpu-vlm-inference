#!/usr/bin/env bash
# 双栈对比实验：Qwen2-VL-2B 双栈对比（vLLM+AWQ 无 offload  vs  HF+NF4）
#
# 设计要点（为什么这么跑）：
# 1. **串行**：一块 4GB 卡只允许一个推理进程，绝不并发。
# 2. **每轮前过基线守卫**：整卡占用 >1000 MiB 直接拒绝——2026-09-18 那次污染环境的
#    教训是"失败结果无法归因"，整轮作废。
# 3. **vLLM 侧必带 VLLM_USE_FLASHINFER_SAMPLER=0**（实测）；
#    max_num_batched_tokens 固定 512（探针实测：默认 1024 时 batch≥2 会死于死法 #2，
#    压到 512 后 KV 池反升到 0.66~0.68 GiB）。
# 4. **两个引擎都跑 batch 1/2/4**：batch=1 看延迟，batch>1 看 vLLM 的批处理优势是否显现。
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
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export VLLM_WSL2_ENABLE_PIN_MEMORY=1

MAX_BASE_MIB="${MAX_BASE_MIB:-1000}"
# vLLM 侧统一配置（经 probe_configs.sh / probe_configs2.sh 实测确定）：
#   gmu 0.80  —— 硬上限：CUDA 侧可用仅 3.21/4.0 GiB，启动检查要求 requested ≤ free → gmu ≤ 0.802
#   offload 0 —— 本次实验的目的（去掉 PCIe 税）
#   mnbt 512  —— **关键**：默认 1024 时 max_num_seqs=2 会把 profiling 激活撑大，
#                KV 池被吃成 −0.17 GiB（死法 #2）；压到 512 后 KV 池反升到 0.66~0.68 GiB，
#                且 batch=1/2/4/8 全部可行、KV 池几乎恒定 → 跨 batch 曲线可比性最好。
VLLM_GMU="${VLLM_GMU:-0.80}"
VLLM_MNBT="${VLLM_MNBT:-512}"

# 【中止式守卫】2026-09-18 教训：vLLM 初始化失败会泄漏显存（进程已死但 WDDM 不回收），
# 后续配置全部会被守卫拦下、白跑一轮。所以一旦基线脏了**立刻终止整个序列**，
# 提示用户重启显卡驱动后再继续，而不是继续空转。
require_clean_gpu() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "$used" -gt "$MAX_BASE_MIB" ]; then
    echo
    echo "🛑 基线守卫：整卡占用 ${used} MiB > 阈值 ${MAX_BASE_MIB} MiB —— 终止整个序列。"
    echo "   原因多半是上一次 vLLM 初始化失败留下的 WDDM 显存泄漏。"
    echo "   处理：按 Win+Ctrl+Shift+B 重启显卡驱动（无效则重启 Windows），再重跑本脚本。"
    echo "   已完成的结果仍有效，重跑时可用 --resume 跳过（见脚本末尾说明）。"
    exit 2
  fi
  echo "[guard] 整卡占用 ${used} MiB ✅"
}

run_one() {
  local engine="$1" tag="$2" batch="$3"

  # --resume：跳过已有有效结果的配置（避免重跑时把已完成的结果覆盖掉）
  if [ "${RESUME:-0}" = "1" ] && [ -s "$RESULTS/${tag}.jsonl" ]; then
    echo
    echo "[skip] $tag 已有结果（$(wc -l < "$RESULTS/${tag}.jsonl") 行），--resume 跳过"
    return 0
  fi

  echo
  echo "=============================================================="
  echo "  [$tag]  engine=$engine  batch=$batch"
  echo "=============================================================="
  require_clean_gpu

  local extra=()
  if [ "$engine" = "vllm" ]; then
    extra=(--gpu-memory-utilization "$VLLM_GMU" --max-num-batched-tokens "$VLLM_MNBT")
  fi

  "$PY" "$SCRIPT_DIR/run_engine.py" \
    --engine "$engine" --batch-size "$batch" --max-base-mib "$MAX_BASE_MIB" \
    "${extra[@]}" \
    --out "$RESULTS/${tag}.jsonl" 2>&1 | tee "$LOGS/${tag}.log" | tail -28
  echo "----- [$tag] 退出码 ${PIPESTATUS[0]} -----"

  # 等显存回落：4GB 卡上"上一轮没释放干净"是最隐蔽的假阴性来源
  sleep 8
  echo "跑完整卡占用：$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}

echo "##############################"
echo "# S1 实验开始 $(date '+%F %T')"
echo "# 模型：Qwen2-VL-2B（HF=bf16 源 + NF4 / vLLM=AWQ 权重，offload=0）"
echo "# vLLM gmu=$VLLM_GMU  mnbt=$VLLM_MNBT  offload=0   HF/NF4 无 offload"
echo "# 子集：$HOME/datasets/OCRBench/subset100/items.jsonl"
echo "# 用法：bash run.sh           全部重跑"
echo "#       RESUME=1 bash run.sh  跳过已有结果（显存泄漏后继续时用）"
echo "##############################"

# 顺序：HF 先跑（无需 EngineCore、失败风险低），vLLM 后跑。
# 若 vLLM 中途失败并泄漏显存，守卫会终止序列，此时用 RESUME=1 重启驱动后续跑。
run_one hf   hf_nf4_b1    1
run_one hf   hf_nf4_b2    2
run_one hf   hf_nf4_b4    4
run_one hf   hf_nf4_b8    8

run_one vllm vllm_awq_b1  1
run_one vllm vllm_awq_b2  2
run_one vllm vllm_awq_b4  4
run_one vllm vllm_awq_b8  8

echo
echo "##############################"
echo "# 实验完成 $(date '+%F %T')"
echo "##############################"
ls -la "$RESULTS"/*.jsonl 2>/dev/null
echo "BENCH_RUN_ALL_DONE"