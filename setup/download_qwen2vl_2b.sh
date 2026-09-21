#!/usr/bin/env bash
# 下载 S1 实验所需的两个模型：Qwen2-VL-2B-Instruct（bf16 源）+ 其 AWQ 量化版
# 走 hf-mirror（huggingface.co 在本机不通）
set -uo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

BIN="$HOME/venvs/vllm/bin"
MODELS="$HOME/models"

download() {
  local repo="$1" dest="$2"
  echo "=============================================================="
  echo "下载 $repo"
  echo "  → $dest"
  date
  echo "=============================================================="
  mkdir -p "$dest"
  if [ -x "$BIN/hf" ]; then
    "$BIN/hf" download "$repo" --local-dir "$dest"
  else
    "$BIN/huggingface-cli" download "$repo" --local-dir "$dest"
  fi
  echo "--- 完成，体积： ---"
  du -sh "$dest" 2>/dev/null
  echo
}

download "Qwen/Qwen2-VL-2B-Instruct"     "$MODELS/Qwen2-VL-2B-Instruct"
download "Qwen/Qwen2-VL-2B-Instruct-AWQ" "$MODELS/Qwen2-VL-2B-Instruct-AWQ"

echo "=============================================================="
echo "全部下载完成 $(date)"
echo "=============================================================="
du -sh "$MODELS"/* 2>/dev/null
echo "DOWNLOAD_ALL_DONE"