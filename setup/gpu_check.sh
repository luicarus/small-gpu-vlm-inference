#!/usr/bin/env bash
# GPU 体检：显存占用 / 残留进程 / 真实算力（判断卡是否可用）
# 用法：bash gpu_check.sh        （纯查询，不占显存）
echo "=== 整卡状态 ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader

echo
echo "=== 可能的残留进程 ==="
ps aux | grep -E "EngineCore|vllm|hf_smoke|vlm_smoke" | grep -v grep || echo "(无残留)"

echo
echo "=== Windows 侧 GPU 进程（WDDM 视角）==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || echo "(查询失败)"
