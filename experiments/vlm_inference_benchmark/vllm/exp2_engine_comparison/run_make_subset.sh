#!/usr/bin/env bash
# 生成 100 条 OCRBench 小图子集（确定性抽样，供两个引擎共用）
set -uo pipefail
PY="$HOME/venvs/vllm/bin/python"
exec "$PY" "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/make_subset.py" "$@"
