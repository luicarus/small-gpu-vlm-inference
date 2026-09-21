#!/bin/bash
set -euo pipefail

export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${VLM_MODEL:-$HOME/models/Qwen2-VL-2B-Instruct-AWQ}"
IMAGE_PATH="${VLM_IMAGE:-$SCRIPT_DIR/../../assets/test_image.png}"
VLLM_BIN="${VLLM_BIN:-$HOME/venvs/vllm/bin/vllm}"
PORT="${VLLM_PORT:-8000}"
LOG_PATH="${VLLM_LOG_PATH:-$HOME/vllm_vlm_serve.log}"
PAYLOAD_PATH="$(mktemp)"

cleanup() {
  rm -f "$PAYLOAD_PATH"
  if [ -n "${SERVE_PID:-}" ] && kill -0 "$SERVE_PID" 2>/dev/null; then
    kill "$SERVE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [ ! -f "$IMAGE_PATH" ]; then
  echo "IMAGE_NOT_FOUND: $IMAGE_PATH"
  exit 1
fi

"$VLLM_BIN" serve "$MODEL" \
  --quantization awq \
  --cpu-offload-gb "${VLM_CPU_OFFLOAD_GB:-1.0}" \
  --max-model-len 1024 \
  --max-num-seqs 1 \
  --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":50176}' \
  --gpu-memory-utilization 0.70 \
  --port "$PORT" >"$LOG_PATH" 2>&1 &
SERVE_PID=$!

ready=0
for i in $(seq 1 150); do
  sleep 2
  if curl -fsS -m 2 "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
    echo "SERVER_READY after $((i * 2))s"
    ready=1
    break
  fi
done

if [ "$ready" -ne 1 ]; then
  echo "SERVER_NOT_READY"
  tail -30 "$LOG_PATH"
  exit 1
fi

"$HOME/venvs/vllm/bin/python" - "$IMAGE_PATH" "$MODEL" "$PAYLOAD_PATH" <<'PY'
import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image

image_path, model, payload_path = sys.argv[1:]
image = Image.open(image_path).convert("RGB")
image.thumbnail((448, 448))
buffer = io.BytesIO()
image.save(buffer, format="JPEG", quality=85)
data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

payload = {
    "model": model,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "请解释这张图中 logical block、block table 和 physical block 的关系。"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }],
    "max_tokens": 64,
    "temperature": 0.0,
}
Path(payload_path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
PY

echo "=== COMPLETION ==="
curl -fsS -m 120 "http://localhost:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data-binary "@$PAYLOAD_PATH"
echo
echo "=== DONE ==="
