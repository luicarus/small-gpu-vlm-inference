"""exp1 扫描器：串行跑一组 vLLM 配置，画出一台机器上的**显存可行边界**。

## 为什么要有这个实验
vLLM 的显存参数不是"越大越好"，而是存在三条互不相同的硬边界。摸不清它们，
就只能靠试错浪费时间；摸清了，任何新模型/新卡都能在几分钟内定位可用区间。

三条边界（死法），错误签名各不相同，必须分开记录：
  #1 预算越界  `is less than desired GPU memory utilization`   ← gmu × 总显存 > 启动时空闲
  #2 KV 池为空 `No available memory for the cache blocks`        ← 预算装不下 权重+激活
  #3 序列太长  `To serve at least one request with the model's max seq len` ← KV 池装不下一条最长序列

## 为什么这么写
1. **一个配置一个子进程**：CUDA 上下文无法干净复位，OOM 还会污染进程状态。
2. **超时杀进程组**：vLLM 的 EngineCore 是独立子进程，只杀父进程会留下孤儿
   继续占着 4GB 显存，把后续所有配置毒化 → 用 start_new_session + killpg 整组带走。
3. **每轮之间等显存回落**：4GB 卡上"上一轮没释放干净"是最隐蔽的假阴性来源。
4. **两条信息合并**：worker 报的（峰值/耗时/成败）+ 引擎日志正则抓的（KV 池 / non_kv 明细），
   后者是解释"为什么死"的关键，但它只在 vLLM 自己的日志里。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.common import RESULTS_ROOT, results_dir_of, write_jsonl  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
WORKER = EXP_DIR / "worker.py"
RESULTS_DIR = results_dir_of(__file__)
LOG_DIR = RESULTS_DIR / "logs"

# ---------------------------------------------------------------------------
# 扫描矩阵：2B 模型（权重 2.74 GiB，offload 可关）
# 与 3B 前置实验最大的差别：**offload 维度消失**（不再被迫把权重放 CPU），
# 于是可以把注意力全部放在 gmu / 序列长度 / 并发这三个真正可调的维度上。
# ---------------------------------------------------------------------------
DEFAULTS = {
    "gpu_memory_utilization": 0.80,
    "cpu_offload_gb": 0.0,
    "max_model_len": 1024,
    "max_num_seqs": 1,
    "max_num_batched_tokens": 0,
    "num_requests": 1,
}

CONFIGS: list[dict] = [
    # --- A 组：gmu 边界（2B 权重小，下界能探得更低）---
    {"label": "A1_gmu55", "gpu_memory_utilization": 0.55, "note": "探 #2 下界：预算 2253 MiB，预计装不下 non_kv"},
    {"label": "A2_gmu60", "gpu_memory_utilization": 0.60, "note": "下界附近"},
    {"label": "A3_gmu65", "gpu_memory_utilization": 0.65, "note": "甜点区下沿"},
    {"label": "A4_gmu70", "gpu_memory_utilization": 0.70, "note": "甜点区中部"},
    {"label": "A5_gmu75", "gpu_memory_utilization": 0.75, "note": "甜点区上沿"},
    {"label": "A6_gmu80", "gpu_memory_utilization": 0.80, "note": "上界：启动检查要求 gmu×4096 ≤ 空闲 3287 → 上限 0.802"},
    {"label": "A7_gmu85", "gpu_memory_utilization": 0.85, "note": "确认 #1：预算 3482 > 空闲 3287"},
    # --- B 组：上下文长度（KV 池能装多长）---
    {"label": "B1_len2048", "max_model_len": 2048, "note": "KV 需求翻倍"},
    {"label": "B2_len4096", "max_model_len": 4096, "note": "视频负载用的档位"},
    {"label": "B3_len8192", "max_model_len": 8192, "note": "预计撞 #3（实测上限约 7440）"},
    # --- C 组：并发（KV 池 + 激活双重压力）---
    {"label": "C1_seqs2", "max_num_seqs": 2, "num_requests": 2,
     "max_num_batched_tokens": 512, "note": "并发 2，mnbt 压到 512"},
    {"label": "C2_seqs4", "max_num_seqs": 4, "num_requests": 4,
     "max_num_batched_tokens": 512, "note": "并发 4"},
    {"label": "C3_seqs8", "max_num_seqs": 8, "num_requests": 8,
     "max_num_batched_tokens": 512, "note": "并发 8"},
    # --- D 组：复核前一组实验的关键发现（mnbt 压小反而让 KV 池更大）---
    {"label": "D1_seqs2_mnbt1024", "max_num_seqs": 2, "num_requests": 2,
     "max_num_batched_tokens": 1024, "note": "对照：mnbt 给 1024，预计 KV 池被激活吃穿（死法 #2）"},
]

DEATH_PATTERNS = [
    ("no_kv", r"No available memory for the cache blocks"),
    ("budget", r"is less than desired GPU memory utilization"),
    ("seq_too_long", r"To serve at least one request with the model's max seq len"),
    ("oom_runtime", r"out of memory"),
]

RE_KV_GIB = re.compile(r"Available KV cache memory:\s*([0-9.]+)\s*GiB")
RE_KV_TOK = re.compile(r"GPU KV cache size:\s*([0-9,]+)\s*tokens")
RE_CONC = re.compile(r"Maximum concurrency for\s*([0-9,]+)\s*tokens per request:\s*([0-9.]+)x")
RE_MEM = re.compile(
    r"Actual usage is\s*([0-9.]+)\s*GiB for consumed memory.*?"
    r"([0-9.]+)\s*GiB for peak activation.*?"
    r"([0-9.]+)\s*GiB for CUDAGraph memory", re.S)
RE_FREE = re.compile(r"Free memory on device \(([0-9.]+)/([0-9.]+) GiB\)")


def classify(log: str) -> str:
    for name, pat in DEATH_PATTERNS:
        if re.search(pat, log):
            return name
    return "ok"


def parse_log(log: str) -> dict:
    """从 vLLM 自身日志里抓出显存账本明细 —— 这是解释"为什么死/活"的关键。"""
    out: dict = {}
    if m := RE_KV_GIB.search(log):
        out["kv_gib"] = float(m.group(1))
    if m := RE_KV_TOK.search(log):
        out["kv_tokens"] = int(m.group(1).replace(",", ""))
    if m := RE_CONC.search(log):
        out["conc_tokens"] = int(m.group(1).replace(",", ""))
        out["conc_x"] = float(m.group(2))
    if m := RE_MEM.search(log):
        out["weights_mib"] = round(float(m.group(1)) * 1024)
        out["activation_mib"] = round(float(m.group(2)) * 1024)
        out["cudagraph_mib"] = round(float(m.group(3)) * 1024)
        out["non_kv_mib"] = out["weights_mib"] + out["activation_mib"] + out["cudagraph_mib"]
    if m := RE_FREE.search(log):
        out["free_gib"] = float(m.group(1))
        out["total_gib"] = float(m.group(2))
    return out


def gpu_used_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        return int(float(out.splitlines()[0]))
    except Exception:                              # noqa: BLE001
        return -1


# 用 CUDA 侧口径测可用量的小探针。
#
# 【为什么两个口径都要查】实测踩过两次，方向相反：
#   ① 只看 nvidia-smi 不够：驱动回收滞后时它已显示 21 MiB，CUDA 侧却还没拿回空间；
#   ② 只看 CUDA 也不够：残留可能完全不在 CUDA 的 `mem_get_info()` 里体现——
#      实测 NVML 显示 687 MiB 被占，而新起的探针进程仍报 3287 MiB 全空。
#      此时 vLLM 的 profiling 会量到偏大的 non_kv（2610 → 3080 MiB），
#      本该成功的配置被判失败（假失败）。
# 所以门槛必须**两个条件同时满足**。
CUDA_FREE_PROBE = """
import torch, json, sys
try:
    free, total = torch.cuda.mem_get_info()
    print(json.dumps({"free_mib": free // 1024**2, "total_mib": total // 1024**2}))
except Exception as exc:
    print(json.dumps({"free_mib": -1, "error": str(exc)[:120]}))
"""


def cuda_free_mib() -> int:
    """用 CUDA 口径读可用显存（与 vLLM 实际能看到的空间更接近）。"""
    try:
        out = subprocess.run([sys.executable, "-c", CUDA_FREE_PROBE],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return int(json.loads(out.splitlines()[-1])["free_mib"])
    except Exception:                              # noqa: BLE001
        return -1


def wait_for_memory_release(min_free_mib: int = 3050, max_nvml_mib: int = 300,
                            timeout_s: int = 90) -> tuple[bool, str]:
    """等显存真正干净：**NVML 与 CUDA 两个口径都要满足**。

    返回 (是否干净, 说明)。任一条件不满足就会一直等到超时——
    这是防"上一轮残留污染下一轮"的关键，比事后解释假失败便宜得多。
    """
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout_s:
        nvml = gpu_used_mib()
        cuda = cuda_free_mib()
        last = f"NVML={nvml} MiB (限 {max_nvml_mib}) / CUDA 可用={cuda} MiB (需 ≥{min_free_mib})"
        if 0 <= nvml <= max_nvml_mib and cuda >= min_free_mib:
            return True, last
        time.sleep(3)
    return False, last


def run_config(cfg: dict, timeout_s: int) -> dict:
    label = cfg["label"]
    params = {**DEFAULTS, **{k: v for k, v in cfg.items() if k not in ("label", "note")}}

    cmd = [sys.executable, str(WORKER),
           "--gpu-memory-utilization", str(params["gpu_memory_utilization"]),
           "--cpu-offload-gb", str(params["cpu_offload_gb"]),
           "--max-model-len", str(params["max_model_len"]),
           "--max-num-seqs", str(params["max_num_seqs"]),
           "--max-num-batched-tokens", str(params["max_num_batched_tokens"]),
           "--num-requests", str(params["num_requests"])]

    base_before = gpu_used_mib()
    cuda_free_before = cuda_free_mib()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{label}.log"

    t0 = time.perf_counter()
    timed_out = False
    # start_new_session=True → 独立进程组；超时后 killpg 能把 EngineCore 一起带走
    with open(log_path, "w", encoding="utf-8") as fp:
        proc = subprocess.Popen(cmd, stdout=fp, stderr=subprocess.STDOUT,
                                start_new_session=True, text=True)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=15)
    wall = round(time.perf_counter() - t0, 2)

    log = log_path.read_text(encoding="utf-8", errors="replace")
    worker_res = {}
    for line in log.splitlines():
        if line.startswith("WORKER_RESULT "):
            try:
                worker_res = json.loads(line[len("WORKER_RESULT "):])
            except json.JSONDecodeError:
                pass

    death = "timeout" if timed_out else classify(log)
    ok = bool(worker_res.get("ok")) and death == "ok"

    rec = {
        "label": label,
        "note": cfg.get("note", ""),
        **params,
        "status": "OK" if ok else "FAIL",
        "death": "" if ok else death,
        "wall_s": wall,
        "base_before_mib": base_before,
        "cuda_free_before_mib": cuda_free_before,
        **{k: v for k, v in worker_res.items() if k != "error"},
        **parse_log(log),
        "error": worker_res.get("error", ""),
        "log": str(log_path),
    }
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeout", type=int, default=600, help="单配置超时（秒）")
    ap.add_argument("--only", default="", help="只跑 label 含该子串的配置")
    ap.add_argument("--dry-run", action="store_true", help="只打印配置矩阵，不占 GPU")
    args = ap.parse_args()

    configs = [c for c in CONFIGS if args.only in c["label"]] if args.only else CONFIGS

    if args.dry_run:
        print(f"exp1 显存边界扫描：{len(configs)} 个配置（GPU 串行执行）")
        for c in configs:
            p = {**DEFAULTS, **{k: v for k, v in c.items() if k not in ("label", "note")}}
            print(f"  - {c['label']:<20} gmu={p['gpu_memory_utilization']} "
                  f"offload={p['cpu_offload_gb']} len={p['max_model_len']} "
                  f"seqs={p['max_num_seqs']} mnbt={p['max_num_batched_tokens']}  # {c['note']}")
        return

    if gpu_used_mib() > 1000:
        print(f"🛑 基线脏（{gpu_used_mib()} MiB），终止。请重启显卡驱动后重跑。", file=sys.stderr)
        sys.exit(2)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp1_sweep_{stamp}.jsonl"

    print(f"exp1 显存边界扫描开始 · {len(configs)} 个配置 · 超时 {args.timeout}s/配置")
    print(f"{'label':<22}{'状态':<6}{'死法':<14}{'峰值MiB':>9}{'KV GiB':>8}{'KV tokens':>11}")
    print("-" * 74)

    records: list[dict] = []
    for i, cfg in enumerate(configs, 1):
        # 【关键门槛】跑之前必须确认 **CUDA 侧**可用量已恢复。
        # 只用 nvidia-smi 不够：驱动回收滞后时它已显示 21 MiB，但 CUDA 仍拿不到全部空间，
        # 会让本该成功的配置量到偏大的 non_kv 而"假失败"（实测 gmu 0.70 就是这样）。
        clean, detail = wait_for_memory_release()
        if not clean:
            print(f"  ⚠️ 显存未真正干净（{detail}）—— 该配置结果可能被污染：{cfg['label']}")

        rec = run_config(cfg, args.timeout)
        records.append(rec)
        print(f"{rec['label']:<22}{rec['status']:<6}{rec['death']:<14}"
              f"{rec.get('peak_mib', 0):>9}{rec.get('kv_gib', 0):>8.2f}"
              f"{rec.get('kv_tokens', 0):>11,}")

    # 全部跑完后也等一次，避免把脏环境留给下一个实验
    wait_for_memory_release()

    meta = {
        "experiment": "exp1_memory_boundary",
        "model": "Qwen2-VL-2B-Instruct-AWQ",
        "n_configs": len(configs),
        "n_ok": sum(1 for r in records if r["status"] == "OK"),
        "n_fail": sum(1 for r in records if r["status"] != "OK"),
        "timeout_s": args.timeout,
        "note": "2B 模型上的显存可行边界；offload 固定 0（权重装得下）",
    }
    write_jsonl(out_path, meta, records)

    print()
    print(f"完成：OK {meta['n_ok']} / FAIL {meta['n_fail']}")
    print(f"结果：{out_path}")
    print("EXP1_SWEEP_OK")


if __name__ == "__main__":
    main()