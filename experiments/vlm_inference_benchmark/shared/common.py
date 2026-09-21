"""vlm_inference_benchmark 共用 Python 工具。

三件事：
1. **路径锚点**：向上找 small-gpu-vlm-inference（认 engines/ + results/），与脚本所在层级解耦。
   为什么不用 `Path(__file__).parents[N]`：脚本一旦被移动（周实验 → benchmark 目录），
   固定层级就错了，而且**错了不会报错，只会把结果写到错地方**。
2. **显存峰值采样**：vLLM 0.28 的推理核心在 EngineCore 子进程里，
   父进程调 `torch.cuda.max_memory_allocated()` 只能看到 ≈0（实测踩过）。
   必须用 NVML 读**整卡**口径，并在父进程开一个采样线程抓峰值。
3. **提示词模板**：图片用 `<|image_pad|>`、视频用 `<|video_pad|>`，两者不能混。
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 路径锚点
# ---------------------------------------------------------------------------

def infra_root(start: Path | None = None) -> Path:
    """向上找到 small-gpu-vlm-inference 根目录（同时含 engines/ 与 results/）。"""
    d = Path(start or __file__).resolve()
    for cand in (d, *d.parents):
        if (cand / "engines").is_dir() and (cand / "results").is_dir():
            return cand
    raise RuntimeError(f"找不到 small-gpu-vlm-inference 根（起点 {d}）")


INFRA_ROOT = infra_root()
BENCH_ROOT = INFRA_ROOT / "experiments" / "vlm_inference_benchmark"
RESULTS_ROOT = INFRA_ROOT / "results" / "vlm_inference_benchmark"
ASSETS_ROOT = INFRA_ROOT / "assets"          # 仓库自带素材（测试图片等）


def results_dir_of(script_file: str | Path) -> Path:
    """由脚本自身位置推出**属于它的**结果目录。

    约定：`results/` 与 `experiments/` 同名同构。
        experiments/vlm_inference_benchmark/vllm/exp1_memory_boundary/sweep.py
        →  results/vlm_inference_benchmark/vllm/exp1_memory_boundary/
    为什么这么设计：实验会随研究推进被**重新分组**（比如把 vLLM 的实验收进 vllm/），
    若在每个脚本里手写 `results/.../exp1_xxx`，一重组就要全文搜索改路径，极易漏。
    镜像规则让分组变化**自动生效**，也保证实验与结果永远对得上。
    """
    exp_dir = Path(script_file).resolve().parent
    rel = exp_dir.relative_to(BENCH_ROOT)
    return RESULTS_ROOT / rel

# 实验统一使用的模型（全部三个实验都用它，保证数据可比）
MODEL_AWQ = str(Path.home() / "models/Qwen2-VL-2B-Instruct-AWQ")   # vLLM 侧
MODEL_BF16 = str(Path.home() / "models/Qwen2-VL-2B-Instruct")      # HF 侧量化源
DEFAULT_IMAGE = ASSETS_ROOT / "test_image.png"

# ---------------------------------------------------------------------------
# 显存峰值采样（NVML 整卡口径）
# ---------------------------------------------------------------------------


class NvmlPeakSampler:
    """后台线程轮询 nvidia-smi，记录整卡显存峰值。

    为什么用 nvidia-smi 而不是 pynvml/torch：
      - `torch.cuda.*` 在父进程读不到子进程（EngineCore）的占用；
      - nvidia-smi 的 `memory.used` 是**整卡**口径，覆盖所有进程，这才是我们关心的
        （4GB 卡上"还剩多少能用"是整卡问题）。
    采样间隔取 100 ms：单次推理在秒级，100 ms 足够抓到峰值，
    又不会让 nvidia-smi 自身的开销干扰测量。
    """

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self.base_mib = 0
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _read_used_mib() -> int:
        out = os.popen(
            "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        ).read().strip()
        try:
            return int(float(out.splitlines()[0]))
        except (ValueError, IndexError):
            return -1

    def start(self) -> int:
        """记录基线并开始采样，返回基线值（MiB）。"""
        self.base_mib = self._read_used_mib()
        self.peak_mib = self.base_mib

        def loop() -> None:
            while not self._stop.is_set():
                v = self._read_used_mib()
                if v > self.peak_mib:
                    self.peak_mib = v
                time.sleep(self.interval_s)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self.base_mib

    def stop(self) -> dict[str, int]:
        """停止采样，返回基线/峰值/净增（MiB）。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return {
            "base_mib": self.base_mib,
            "peak_mib": self.peak_mib,
            "delta_mib": self.peak_mib - self.base_mib,
        }


# ---------------------------------------------------------------------------
# 提示词模板
# ---------------------------------------------------------------------------

def build_image_prompt(question: str) -> str:
    """Qwen2-VL 对话模板 + **单图**占位符。"""
    return (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_video_prompt(question: str) -> str:
    """Qwen2-VL 对话模板 + **视频**占位符。

    与图片的唯一差别就是 `<|video_pad|>`（源码 `model_executor/models/qwen2_vl.py:1279`）。
    写错不会报错，只会让模型看到一堆无意义的 pad token —— 所以模板必须集中在这里。
    """
    return (
        "<|im_start|>user\n"
        "<|vision_start|><|video_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# 结果落盘
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, meta: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """写「首行 meta + 逐条记录」的 JSONL（全项目统一的落盘格式）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """读回 (meta, rows)。"""
    with open(path, encoding="utf-8") as fp:
        lines = [json.loads(l) for l in fp if l.strip()]
    return lines[0]["_meta"], lines[1:]