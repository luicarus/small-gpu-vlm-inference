#!/usr/bin/env bash
# 校验博客要引用的两处源码结论（offload 只包解码层 / vision tower 不经过 offloader）
SP=~/venvs/vllm/lib/python3.12/site-packages/vllm

echo "===== 1. get_offloader().wrap_modules 调用点 ====="
grep -rn "get_offloader()" "$SP/model_executor/models/utils.py" | head
echo "--- 上下文 ---"
grep -n -B3 -A6 "get_offloader()" "$SP/model_executor/models/utils.py" | head -40

echo
echo "===== 2. qwen2.py make_layers ====="
grep -n "make_layers" "$SP/model_executor/models/qwen2.py" | head

echo
echo "===== 3. qwen2_5_vl vision tower 是否用普通 ModuleList ====="
grep -n "self.blocks" "$SP/model_executor/models/qwen2_5_vl.py" | head

echo
echo "===== 4. offload 硬上限日志行 ====="
grep -n "Total CPU offloaded parameters" "$SP/model_executor/offloader/uva.py"

echo
echo "===== 5. gmu 预算校验（#1 闸门）====="
grep -n -B4 -A8 "is less than desired" "$SP/v1/worker/utils.py"

echo
echo "===== 6. #2 / #3 闸门 ====="
grep -n -B6 -A6 "No available memory for the cache blocks" "$SP/v1/core/kv_cache_utils.py"
grep -n -B4 -A8 "To serve at least one request" "$SP/v1/core/kv_cache_utils.py"
