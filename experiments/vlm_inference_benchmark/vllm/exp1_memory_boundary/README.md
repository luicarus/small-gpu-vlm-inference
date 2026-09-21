# exp1_memory_boundary

扫描 vLLM 的显存相关参数，找出**可用区间**与**越界时的失败方式**。

## 怎么跑

```bash
bash run.sh                 # 全部 14 个配置，约 20 分钟
bash run.sh --dry-run       # 只看配置矩阵，不占 GPU
bash run.sh --only A        # 只跑 A 组（gmu 边界）
bash run.sh --timeout 900   # 单配置超时放宽
```

## 文件

| 文件 | 作用 |
|---|---|
| `sweep.py` | 扫描编排：配置矩阵、子进程隔离、超时 killpg、失败分类、显存双口径门槛 |
| `worker.py` | 单配置执行器：起一次 vLLM、跑几条请求、输出一行 JSON |
| `summarize.py` | 把结果整理成报告用表（只打印，不落盘） |
| `run.sh` | 入口 |

## 参数

固定的：`offload=0`（2B 权重 2.74 GiB 装得下，这是本实验成立的前提）、`max_pixels=262144`。

被扫描的：`gpu_memory_utilization`(0.55~0.85) · `max_model_len`(1024~8192) ·
`max_num_seqs`(1~8) · `max_num_batched_tokens`(auto/512/1024)。

## 输出

`results/vlm_inference_benchmark/vllm/exp1_memory_boundary/`

- `exp1_sweep_<时间戳>.jsonl` —— 首行是该批次的 meta，之后每行一个配置。
  每条记录含：配置参数、状态、失败类型（`death`）、峰值显存、
  **从 vLLM 日志里解析出的显存账本**（`non_kv_mib` / `kv_gib` / `kv_tokens` / `conc_x`）、
  以及跑前的 NVML 与 CUDA 两个口径的基线。
- `logs/<label>.log` —— 每个配置的完整引擎输出，失败时靠它定位原因。

## 怎么读结果

**三个可以手算的量**：

```
non_kv  = 权重 + profiling 峰值激活 + CUDA graph      （本机 2610 MiB，与 gmu 无关）
KV 池   = gmu × 总显存 − non_kv
最长序列 = KV 池 ÷ 28.1 KiB/token
```

**`death` 字段的三种取值**（对应后面 `error` 里的报错原文）：

| `death` | 含义 |
|---|---|
| `budget` | `gmu × 总显存` 超过了启动时的空闲显存 |
| `no_kv` | 预算装不下 `non_kv`，KV 池为负 |
| `seq_too_long` | KV 池存在，但装不下一条 `max_model_len` 的序列 |

**⚠️ 读数据前先看 `cuda_free_before_mib` 与 `base_before_mib`**：
两个口径都干净时结果才可信。如果某条记录的 `non_kv_mib` 明显偏离 2610 MiB（例如 3080），
那次运行的环境有问题，应当重跑该配置而不是解释它。这个坑的完整记录见 `REPORT.md` §5.2。

## 相关

- 机制解释与完整数据：`REPORT.md` §2
- 被取代的 3B 版本：`../preliminary_3b_memory_sweep/`
