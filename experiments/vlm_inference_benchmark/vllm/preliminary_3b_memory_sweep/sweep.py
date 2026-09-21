"""前置实验参数扫描执行器：逐配置串行跑 engines/vllm/vlm_smoke.py，记录显存/KV/吞吐/死法。

设计要点（为什么这么写）：
1. **一个配置一个子进程**：CUDA 上下文和显存池无法在同一进程里干净复位，
   OOM 还会污染进程状态；子进程隔离保证每个配置都是"干净开机"。
2. **复用 vlm_smoke.py 当执行器**：引擎调用逻辑只写一份，扫描器只管
   调度 + 解析 + 记账，避免两套代码跑出不一致的结果。
3. **超时必须杀进程组**：vLLM 的 EngineCore 跑在子进程里，只杀父进程会留下
   孤儿进程占着 4GB 显存，把后续配置全部毒化。所以用 start_new_session
   建独立进程组，超时后 killpg 整组带走。
4. **每个配置跑完等显存回落**：4GB 卡上"上一轮没释放干净"是最隐蔽的假阴性来源，
   跑下一个之前先轮询确认占用回到基线附近。
5. **死法分类**：vLLM 三种启动失败的错误签名各不相同，必须分开记，
   否则"全红了"看不出边界在哪。
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
from datetime import datetime
from pathlib import Path

# 路径与结果目录统一走 shared（现在本目录已在 vlm_inference_benchmark/ 内）。
# 为什么不用手写层级：本目录搬过两次（w4_memory_sweep → preliminary_3b_memory_sweep
# → vlm_inference_benchmark/vllm/preliminary_3b_memory_sweep），每次手写层级都要全文改；
# 用镜像规则后，以后再分组也不会断。
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.common import INFRA_ROOT as INFRA_DIR  # noqa: E402
from shared.common import results_dir_of  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parent
SMOKE_TEST = INFRA_DIR / "engines" / "vllm" / "vlm_smoke.py"
IMAGE = INFRA_DIR.parent / "docs" / "assets" / "test_image.png"
RESULTS_DIR = results_dir_of(__file__)
LOG_DIR = RESULTS_DIR / "logs"

# ---------------------------------------------------------------------------
# 扫描矩阵：围绕实测边界设计（自由 3287 MiB / non_kv 2591 MiB / KV 36 KiB per token）
# ---------------------------------------------------------------------------
CONFIGS: list[dict] = [
    # --- A 组：gmu 边界扫描（offload 固定 1.0，len=1024，seqs=1）---
    {"label": "01_baseline", "gpu_memory_utilization": 0.70, "cpu_offload_gb": 1.0,
     "note": "对照基线（已验证可跑）"},
    {"label": "02_gmu60", "gpu_memory_utilization": 0.60, "cpu_offload_gb": 1.0,
     "note": "探 #2 下界：预算预计装不下权重"},
    {"label": "03_gmu65", "gpu_memory_utilization": 0.65, "cpu_offload_gb": 1.0,
     "note": "探甜点区下沿（KV 池预计 ~71 MiB）"},
    {"label": "04_gmu75", "gpu_memory_utilization": 0.75, "cpu_offload_gb": 1.0,
     "note": "甜点区中部"},
    {"label": "05_gmu80", "gpu_memory_utilization": 0.80, "cpu_offload_gb": 1.0,
     "note": "探 #1 上沿：预算 3277 vs 空闲 3287，仅差 10 MiB（擦边）"},
    {"label": "06_gmu85", "gpu_memory_utilization": 0.85, "cpu_offload_gb": 1.0,
     "note": "确认 #1：预算 3482 > 空闲"},
    # --- B 组：offload 维度（gmu 固定 0.70）---
    {"label": "07_offload0", "gpu_memory_utilization": 0.70, "cpu_offload_gb": 0.0,
     "note": "权重全进显存，预计撞 #2"},
    {"label": "08_offload2", "gpu_memory_utilization": 0.70, "cpu_offload_gb": 2.0,
     "note": "offload 换 KV：预计 KV 池 ~1300 MiB"},
    # --- C 组：上下文与并发维度 ---
    {"label": "09_len2048", "gpu_memory_utilization": 0.70, "cpu_offload_gb": 1.0,
     "max_model_len": 2048, "note": "KV 需求翻倍（72 MiB），预计仍稳"},
    {"label": "10_seqs4", "gpu_memory_utilization": 0.70, "cpu_offload_gb": 1.0,
     "num_requests": 4, "max_num_seqs": 4, "note": "并发验证：4 条不同提问"},
    # --- D 组：解药验证（chunked prefill 解耦 max_num_batched_tokens）---
    # 09/10 的失败根因是激活值随 max_num_batched_tokens 膨胀（1024→2048→4096），
    # 该值由 vLLM 从 max_model_len × max_num_seqs 自动推导。显式限小它，看能否救活。
    {"label": "11_len2048_mnbt1024", "gpu_memory_utilization": 0.70, "cpu_offload_gb": 1.0,
     "max_model_len": 2048, "max_num_batched_tokens": 1024,
     "note": "解药验证：len 2048 但把 batched tokens 压回 1024"},
    {"label": "12_seqs4_mnbt1024", "gpu_memory_utilization": 0.70, "cpu_offload_gb": 1.0,
     "num_requests": 4, "max_num_seqs": 4, "max_num_batched_tokens": 1024,
     "note": "解药验证：seqs 4 但把 batched tokens 压回 1024"},
]

DEFAULTS = {"max_model_len": 1024, "max_num_seqs": 1, "num_requests": 1,
            "max_num_batched_tokens": 0}

# 输出解析：冒烟测试打印的指标 + vLLM 自己打印的关键行
PATTERNS: dict[str, tuple[str, type]] = {
    "base_mib": (r"GPU_MEM_BASE_MIB:\s*(\d+)", int),
    "peak_mib": (r"GPU_MEM_PEAK_MIB:\s*(\d+)", int),
    "delta_mib": (r"GPU_MEM_PEAK_DELTA_MIB:\s*(-?\d+)", int),
    "gen_time_s": (r"GEN_TIME_S:\s*([\d.]+)", float),
    "output_tokens": (r"OUTPUT_TOKENS:\s*(\d+)", int),
    "tokens_per_s": (r"TOKENS_PER_S:\s*([\d.]+)", float),
    "kv_cache_gib": (r"Available KV cache memory:\s*([\d.]+)\s*GiB", float),
    "kv_cache_tokens": (r"GPU KV cache size:\s*([\d,]+)\s*tokens", lambda s: int(s.replace(",", ""))),
    "max_concurrency": (r"Maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x", float),
    "cpu_offloaded_gib": (r"Total CPU offloaded parameters:\s*([\d.]+)", float),
    "model_load_gib": (r"Model loading took ([\d.]+) GiB memory", float),
    "model_load_s": (r"Model loading took [\d.]+ GiB memory and ([\d.]+) seconds", float),
}

# 死法分类：按 vLLM 源码里三处 raise 的原文匹配
DEATH_SIGNATURES = [
    ("#1_budget", "is less than desired GPU memory utilization"),
    ("#2_kv_zero", "No available memory for the cache blocks"),
    ("#3_seq_too_long", "To serve at least one request with the model's max seq len"),
    ("oom_runtime", "out of memory"),
]


def gpu_used_mib() -> float | None:
    """整卡已用显存（NVML 口径，母进程不建 CUDA context）。"""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024
    except Exception:
        return None


def wait_for_gpu_release(baseline: float | None, timeout_s: float = 60.0) -> None:
    """等上一轮进程真正释放显存——4GB 卡上残留占用会毒化下一个配置。"""
    if baseline is None:
        return
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        now = gpu_used_mib()
        if now is None or now <= baseline + 200:  # 留 200 MiB 容差（桌面会浮动）
            return
        time.sleep(2.0)
    # 注意：临跑前再取一次可能又变成 None，格式化前必须判空，否则扫描会被打断
    now = gpu_used_mib()
    shown = f"{now:.0f} MiB" if now is not None else "N/A"
    print(f"  [warn] 显存未回落到基线附近（当前 {shown}），继续但结果可能偏低")


def run_config(cfg: dict, timeout_s: float) -> dict:
    label = cfg["label"]
    params = {**DEFAULTS, **{k: v for k, v in cfg.items() if k != "note"}}
    cmd = [
        sys.executable, str(SMOKE_TEST),
        "--image", str(IMAGE),
        "--gpu-memory-utilization", str(params["gpu_memory_utilization"]),
        "--cpu-offload-gb", str(params["cpu_offload_gb"]),
        "--max-model-len", str(params["max_model_len"]),
        "--max-num-seqs", str(params["max_num_seqs"]),
        "--num-requests", str(params["num_requests"]),
    ]
    if params.get("max_num_batched_tokens"):
        cmd += ["--max-num-batched-tokens", str(params["max_num_batched_tokens"])]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{label}.log"

    record = {"label": label, "note": cfg.get("note", ""), "params": params,
              "started_at": datetime.now().isoformat(timespec="seconds"),
              "log": str(log_path.relative_to(INFRA_DIR.parent))}

    print(f"  → {label}: gmu={params['gpu_memory_utilization']} "
          f"offload={params['cpu_offload_gb']} len={params['max_model_len']} "
          f"seqs={params['max_num_seqs']} reqs={params['num_requests']}")

    t0 = time.perf_counter()
    timed_out = False
    with open(log_path, "w", encoding="utf-8") as log_fp:
        # start_new_session=True：让子进程及其 EngineCore 后代同属一个进程组，
        # 超时才能一锅端（见文件头设计要点 3）
        proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    record["wall_time_s"] = round(time.perf_counter() - t0, 1)

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    record["exit_code"] = proc.returncode

    for key, (pattern, cast) in PATTERNS.items():
        m = re.search(pattern, log_text)
        if m:
            try:
                record[key] = cast(m.group(1))
            except (ValueError, TypeError):
                pass

    if timed_out:
        record["status"] = "TIMEOUT"
    elif "VLM_SMOKE_TEST_OK" in log_text:
        record["status"] = "OK"
    else:
        record["status"] = "FAIL"
        record["death"] = "unknown"
        lowered = log_text.lower()
        for name, signature in DEATH_SIGNATURES:
            if signature.lower() in lowered:
                record["death"] = name
                break
        # 抓最后几行非空日志，便于人工判断
        tail = [ln for ln in log_text.strip().splitlines() if ln.strip()][-6:]
        record["log_tail"] = " | ".join(tail)[:800]

    return record


def summarize(records: list[dict]) -> str:
    header = ("| 配置 | gmu | offload | len | seqs | 跑前基线 MiB | 状态 | 死法 | 峰值 MiB | "
              "KV池 GiB | KV tokens | 并发 | tok/s |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for r in records:
        p = r["params"]
        base = r.get("base_before_mib")
        base_cell = "—" if base is None else (f"⚠️ {base}" if r.get("contaminated") else str(base))
        rows.append(
            f"| {r['label']} | {p['gpu_memory_utilization']} | {p['cpu_offload_gb']} | "
            f"{p['max_model_len']} | {p['max_num_seqs']} | {base_cell} | {r['status']} | "
            f"{r.get('death', '—')} | {r.get('peak_mib', '—')} | "
            f"{r.get('kv_cache_gib', '—')} | {r.get('kv_cache_tokens', '—')} | "
            f"{r.get('max_concurrency', '—')} | {r.get('tokens_per_s', '—')} |"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="",
                        help="只跑指定 label，逗号分隔（如 01_baseline,02_gmu60）")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="单配置超时秒数（默认 600）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不跑")
    # 【闸门语义修正 2026-09-15】最初的假设是"桌面占用会抬高 non_kv、吃掉 KV 池"——
    # 实测推翻了它：base=803 MiB 时跑基线，KV 池仍是 0.25 GiB / 7248 token，与 base=134 完全一致。
    # 原因：vLLM 的 profiling 测的是"跑 forward 前后的空闲显存差"，别人的既有占用会被差值约掉。
    # 所以闸门只用于兜底"上一轮实验残留/别的实验在跑"这类异常，不该对正常桌面占用报警。
    parser.add_argument("--max-base-mib", type=float, default=1500.0,
                        help="兜底闸门：跑前整卡占用超过此值（MiB）视为环境异常（默认 1500）")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="环境异常时仍然继续跑（记录会被标记 contaminated=true）")
    args = parser.parse_args()

    selected = CONFIGS
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        selected = [c for c in CONFIGS if c["label"] in wanted]
        if not selected:
            print(f"没有匹配的 label：{wanted}")
            sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = RESULTS_DIR / f"w4_sweep_{stamp}.jsonl"

    print(f"显存边界扫描：{len(selected)} 个配置（GPU 串行执行）")
    if args.dry_run:
        for c in selected:
            p = {**DEFAULTS, **c}
            print(f"  - {c['label']}: gmu={p['gpu_memory_utilization']} "
                  f"offload={p['cpu_offload_gb']} len={p['max_model_len']} "
                  f"seqs={p['max_num_seqs']} reqs={p['num_requests']}  # {c.get('note', '')}")
        return

    baseline = gpu_used_mib()
    print(f"开跑前整卡占用：{baseline:.0f} MiB" if baseline else "开跑前整卡占用：N/A")
    records: list[dict] = []
    for i, cfg in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {cfg['label']}")

        # 环境闸门：跑前量基线，脏了就拒绝（见 --max-base-mib 注释）
        base_before = gpu_used_mib()
        dirty = base_before is not None and base_before > args.max_base_mib
        if dirty and not args.allow_dirty:
            print(f"  ✗ 环境不干净：跑前整卡占用 {base_before:.0f} MiB > 上限 "
                  f"{args.max_base_mib:.0f} MiB —— 拒绝出数据。\n"
                  f"    处置：关掉占用 GPU 的程序（浏览器/视频等）后重跑，"
                  f"或确认要带脏跑时加 --allow-dirty（结果会标记 contaminated）")
            break

        record = run_config(cfg, args.timeout)
        record["base_before_mib"] = round(base_before) if base_before is not None else None
        record["contaminated"] = bool(dirty)
        records.append(record)
        line = f"  ← {record['status']}"
        if record.get("death"):
            line += f" ({record['death']})"
        for key, unit in (("peak_mib", "MiB"), ("kv_cache_gib", "GiB"),
                          ("tokens_per_s", "tok/s"), ("wall_time_s", "s")):
            if key in record:
                line += f" {key}={record[key]}{unit}"
        print(line)
        with open(jsonl_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        wait_for_gpu_release(baseline)

    table = summarize(records)
    print("\n" + table)
    # 说明：这里只打印、不落盘汇总 .md —— 结果文档统一只保留一份
    # 避免每次扫描都堆一个 summary 文件。
    print(f"\n原始记录：{jsonl_path}")
    print("提示：跨批次合并总表用 `python experiments/preliminary_3b_memory_sweep/summarize.py`")


if __name__ == "__main__":
    main()
