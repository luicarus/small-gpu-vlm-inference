#!/usr/bin/env bash
# SGLang 版 exp3：帧数扫描（共享前缀长度对加速比的影响）
#
# 与 vLLM 版（`../vllm/exp3_prefix_caching/run_frames.sh`）**共用同一份 workload**，
# 目标是同口径对比两种前缀复用机制（radix 树 vs 块哈希链）。
#
# ## 帧数档位与 context_length 的对应
# 每帧约 161 token（实测：8 帧 1296 / 16 帧 2576 / 32 帧 5152 / 40 帧 6448）。
# SGLang 要求 context_length ≥ 前缀 + 问题 + 输出，否则报错。所以：
#   8 帧  → 2048
#   16 帧 → 4096
#   32 帧 → 6144
#   40 帧 → 8192
#
# ## 显存前提（实测）
# mem_fraction_static=0.90 → KV 池 14,274 token，可装下 40 帧的 6,448 token 前缀。
# ⚠️ 注意 mfs 语义与 vLLM 的 gmu **相反**：越大 → slack 越小 → KV 越大。
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../shared/paths.sh"
bench_init "${BASH_SOURCE[0]}"

PY="${SGLANG_PYTHON_BIN:-$HOME/venvs/sglang/bin/python}"
RESULTS="$RESULTS_ROOT/sglang"
LOGS="$RESULTS/logs"
mkdir -p "$LOGS"

# flashinfer JIT 需要 CUDA 12.1+ 的编译选项，而本机系统 nvcc 是 12.0；
# 指到 venv 自带的 CUDA 13.4（详见 run_smoke.sh 的说明）
VENV_CUDA="$(dirname "$PY")/../lib/python3.12/site-packages/nvidia/cu13"
[ -d "$VENV_CUDA" ] && { export CUDA_HOME="$VENV_CUDA"; export CUDA_PATH="$VENV_CUDA"; export PATH="$VENV_CUDA/bin:$PATH"; }

# 基线门槛：收紧到 200 MiB。
#
# 为什么不是 800：本机**干净基线一直稳定在 ~30 MiB**（实测多次），
# 而 800 的阈值放过了 frames32_cacheoff 之后留下的 **679 MiB 残留**——
# 那与 exp1 踩过的 687 MiB 是**同一个坑**（见 REPORT.md §5.2）：
# 残留会抬高后续配置的基线，制造"假失败"或污染性能数据。
# 经验值门槛不如"照着干净基线设"可靠。
MAX_BASE_MIB="${MAX_BASE_MIB:-200}"
N="${N:-10}"
MFS="${MFS:-0.90}"
FRAME_CONFIGS="${FRAME_CONFIGS:-8:2048:0.90 16:4096:0.90 32:6144:0.92 40:8192:0.93}"
# attention backend：默认 triton。原因见 run_smoke.sh —— 本机系统 CUDA 12.0 与
# SGLang 需要的 flashinfer 存在版本差，只能用 triton 这类不依赖 flashinfer 的后端。
# ⚠️ 实测同一负载下 backend 差异极大（triton 1.00× / torch_native 2.34× / flex_attention 4.01×），
# 所以**必须固定住**，否则组间不可比。
ATTN_BACKEND="${ATTN_BACKEND:-triton}"

FAILED=0

run_one() {
  local cache="$1" frames="$2" ctx="$3" mfs="${4:-$MFS}"
  local tag="frames${frames}_cache${cache}"
  echo
  echo "=============================================================="
  echo "  [$tag]  frames=$frames  context_len=$ctx  caching=$cache  N=$N  mfs=$mfs"
  echo "=============================================================="

  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "$used" -gt "$MAX_BASE_MIB" ]; then
    echo "🛑 基线守卫：${used} MiB > ${MAX_BASE_MIB} —— 终止（残留未回收）"
    exit 2
  fi
  echo "[guard] 整卡 ${used} MiB ✅"

  local extra=()
  [ "$cache" = "off" ] && extra=(--disable-radix-cache)

  "$PY" "$SCRIPT_DIR/run_cache.py" \
    --num-questions "$N" --frames "$frames" --context-length "$ctx" \
    --mem-fraction-static "$mfs" --attention-backend "$ATTN_BACKEND" "${extra[@]}" \
    --out "$RESULTS/${tag}.jsonl" > "$LOGS/${tag}.log" 2>&1
  local rc=$?

  if [ $rc -eq 0 ]; then
    grep -E '冷启动|后续请求|平均命中率|整轮墙钟|SGLANG_RUN_OK' "$LOGS/${tag}.log" | sed 's/^/  /'
  else
    echo "  ⚠️ 该组失败（rc=$rc，见 $LOGS/${tag}.log）"
    grep -vE 'torchcodec|libavutil|FFmpeg|_dlopen|ctypes|load_library|core_library_path|load_torchcodec|for each of those versions' \
      "$LOGS/${tag}.log" | grep -iE 'Error|error|Invalid|not support' | head -3 | sed 's/^/    /'
    FAILED=$((FAILED+1))
  fi

  sleep 8
  echo "  跑完整卡占用：$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}

echo "##############################"
echo "# SGLang exp3 帧数扫描  $(date '+%F %T')"
echo "# 负载：与 vLLM 版共用同一份 workload.json（同一张图重复成帧）"
echo "# 档位（frames:context_len:mem_fraction_static）：$FRAME_CONFIGS"
echo "##############################"

# 每档的 mfs 可单独给：`frames:ctx:mfs`
# 为什么需要：**长 context 会抬高 profiling 阶段的 activation 峰值**，
# 于是 `available_gpu_memory` 变小、`1 - available/pre` 变大 → 最小可用 mfs 变大。
# 实测 ctx=6144 时 mfs=0.90 直接启动失败（SGLang 报 minimum viable = 0.8982）。
for cfg in $FRAME_CONFIGS; do
  frames="$(echo "$cfg" | cut -d: -f1)"
  ctx="$(echo "$cfg" | cut -d: -f2)"
  mfs="$(echo "$cfg" | cut -d: -f3)"
  [ -z "$mfs" ] && mfs="$MFS"
  run_one off "$frames" "$ctx" "$mfs"
  run_one on  "$frames" "$ctx" "$mfs"
done

echo
echo "##############################"
echo "# 完成 $(date '+%F %T')  失败 $FAILED 组"
echo "##############################"
ls -la "$RESULTS"/frames*.jsonl 2>/dev/null
echo "SGLANG_FRAMES_DONE"