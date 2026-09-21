#!/usr/bin/env bash
# A 实验前置探测：更大的帧数需要更大的 max_model_len，而这会挤压 KV 池。
# 先探最大档（48 帧），看 ① 能不能起来 ② KV 池装得下共享前缀吗 ③ 会不会 OOM。
set -uo pipefail

# 路径锚点：向上找 small-gpu-vlm-inference，与脚本所在层级解耦（见 shared/paths.sh 说明）
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../shared" && pwd)/paths.sh"
bench_init "${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${VLLM_PYTHON_BIN:-$HOME/venvs/vllm/bin/python}"
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
export VLLM_WSL2_ENABLE_PIN_MEMORY=1 HF_HUB_DISABLE_XET=1

FRAMES="${1:-48}"
MML="${2:-8192}"
LOG="/tmp/w6_probe_f${FRAMES}.log"

echo "探测：frames=$FRAMES  max_model_len=$MML"
echo "（每帧约 162 token，预计共享前缀 ≈ $((FRAMES * 162)) token）"
echo

"$PY" "$SCRIPT_DIR/run_cache.py" \
  --num-questions 2 --frames "$FRAMES" --max-model-len "$MML" \
  --out "/tmp/w6_probe_f${FRAMES}.jsonl" >"$LOG" 2>&1
rc=$?

echo "=== 引擎显存账本 ==="
grep -oE 'Available KV cache memory: [0-9.-]+ GiB|GPU KV cache size: [0-9,]+ tokens|Maximum concurrency[^,]*' "$LOG" | head -4

echo
echo "=== 关键行 ==="
grep -E 'ValueError|No available memory|is less than desired|out of memory' "$LOG" | head -3 || echo "  ✅ 无启动失败"

echo
echo "=== 请求结果 ==="
grep -E '^  [0-9]+ +[0-9]+ +[0-9]+' "$LOG" | head -4 || tail -12 "$LOG"

echo
if [ $rc -eq 0 ]; then echo "✅ frames=$FRAMES 可行"; else echo "❌ frames=$FRAMES 失败（rc=$rc）"; fi
echo "PROBE_DONE"