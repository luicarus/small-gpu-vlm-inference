#!/usr/bin/env bash
# exp1 入口：显存可行边界扫描
#
# 用法：
#   bash run.sh                全部 14 个配置（约 20 分钟）
#   bash run.sh --dry-run      只看配置矩阵（不占 GPU）
#   bash run.sh --only A        只跑 A 组
#   bash run.sh --timeout 900   单配置超时放宽到 15 分钟
set -uo pipefail

# 路径锚点：向上找 small-gpu-vlm-inference，与脚本所在层级解耦（见 shared/paths.sh 说明）
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../shared" && pwd)/paths.sh"
bench_init "${BASH_SOURCE[0]}"

PY="$(bench_python)"

require_clean_gpu || exit 2

# 用镜像规则定位自己的脚本与结果目录，避免在脚本里手写分组层级
EXP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$PY" "$EXP_DIR/sweep.py" "$@"