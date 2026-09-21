# exp2_engine_comparison

在**关闭 CPU offload** 的前提下，对比 `vLLM + AWQ` 与 `transformers + bitsandbytes NF4`。

## 怎么跑

```bash
bash run_make_subset.sh     # 生成 OCRBench 100 条小图子集（seed=42）
bash run_smoke.sh           # 两个引擎各几条，确认环境没坏
bash run.sh                 # 正式实验：两引擎 × batch 1/2/4/8，约 10 分钟
RESUME=1 bash run.sh        # 跳过已有结果（显存异常后续跑）
```

配置探针（找出 batch≥2 可行的参数组合）：

```bash
bash probe_configs.sh
bash probe_configs2.sh
```

## 文件

| 文件 | 作用 |
|---|---|
| `make_subset.py` | 从 OCRBench 抽 100 条**小图**子集，确定性抽样 |
| `run_engine.py` | 双引擎执行器：模型类自动分发、NVML 峰值采样、基线守卫 |
| `compare.py` | 四指标分析 + 跨引擎一致率（严格 / 语义双口径） |
| `check_jit.sh` | 检查日志里的 Triton 运行时 JIT 警告 + 逐条延迟汇总 |
| `probe_configs*.sh` | 配置探针（用单图冒烟快速试，不跑满 100 条） |

## 参数

`gpu_memory_utilization=0.80`（硬上限 0.802）· `cpu_offload_gb=0` ·
`max_num_batched_tokens=512`（**batch≥2 必须**，否则激活吃穿 KV 池）· `max_pixels=262144` ·
`temperature=0`（贪心）· `max_tokens=32`。

## 输出

`results/vlm_inference_benchmark/vllm/exp2_engine_comparison/`

- `<引擎>_b<batch>.jsonl`（8 份）—— 首行 meta（运行时长、峰值显存、加载时间等），
  之后每行一条：问题、真值、输出、单条延迟。
- `logs/<配置>.log` —— 引擎完整输出。

## 怎么读结果

```bash
python compare.py                       # 四指标总表 + 延迟分位 + 准确率 + 一致率
python ../../../tools/peek_results.py results/.../hf_nf4_b1.jsonl --wrong   # 看答错的
```

**报什么**：运行时长、吞吐、**峰值显存**、P50/P99、加载时间、准确率。

**⚠️ 这个实验的边界**（结论不能超出这些）：

- 两个栈**量化格式不同**（AWQ 预量化 vs NF4 运行时量化），
  且**只有 HF 一侧量化了视觉塔**（AWQ 的官方默认是不量化 `visual`）。
  所以这是**工程对比**，不是纯引擎对比。显存结论要打折（HF 的优势被放大约 1.3 GiB），
  速度结论受影响较小。
- 准确率只作回归检查：n=100 分辨不出 1~2 个百分点的差异。
- 跨引擎**一致率的严格口径是陷阱**：46% 看起来很低，但人工检视发现差异绝大多数是
  表述风格（`says "X"` vs `reads "X"`），语义口径是 56%。

## 相关

- 完整分析与适用边界：`REPORT.md` §3
