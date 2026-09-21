# 前置实验：3B 模型上的显存边界扫描

> ⚠️ **这是一组被取代的前置实验，保留作方法学记录，不是当前结论的来源。**
> 当前结论请看 [`../vlm_inference_benchmark/`](../vlm_inference_benchmark/)（同一台机器、2B 模型）。

## 为什么它会被取代

最初在 **Qwen2.5-VL-3B** 上做显存边界扫描。扫完之后发现一个致命问题：

> 3B 的 AWQ 权重是 **3.17 GiB**，在 4GB 卡上**装不下**（可用的 CUDA 空间只有 3.21 GiB，
> 还要留给激活值和 KV cache）。于是 vLLM 被迫开启 `cpu_offload_gb`，
> 权重放在 CPU、解码时每个 token 都要经 PCIe 搬运。

这带来两个后果：

1. **测出的性能差距无法归因**——你测到的是"谁被迫 offload"的差距，不是引擎本身的差距。
   PCIe 有效带宽约 7 GB/s，显存约 192 GB/s，**差一个数量级**，足以掩盖其它所有因素。
2. **`offload` 维度本身成了噪声**——它不是一个"可选的优化"，而是被显存逼出来的妥协，
   把它当作自变量扫描，扫出来的曲线解释不了任何工程决策。

换成 **2B**（AWQ 权重 2.74 GiB）后 `offload=0` 装得下，这个变量才被真正消掉。

## 它仍然有价值的地方

这组实验是**发现上述问题的过程本身**，并且留下了三条可复用的方法论结论
（都已在当前实验里沿用）：

| 发现 | 含义 |
|---|---|
| **三种启动失败模式**各有不同错误签名 | `#1 预算越界` / `#2 KV 池为空` / `#3 序列太长`；混在一起看就只会看到"全红了" |
| **`gpu_memory_utilization` 的硬上限是机器常数** | 上限 = CUDA 可用 ÷ 总显存 = 3.21/4.0 = **0.802**，与模型无关；设 0.85 以上必然死于 #1 |
| **`max_num_batched_tokens` 会吃掉 KV 池** | 它决定 profiling 那次 forward 的激活峰值；不显式压小，`max_model_len`/`max_num_seqs` 一调大就死于 #2 |

这三点在 2B 上**完全复现**（见 `../vlm_inference_benchmark/exp1_memory_boundary/`），
说明它们描述的是 vLLM 的机制，而不是某个模型的特性。

## 为什么不可复现

模型文件**已删除**（释放 10.3 GiB 磁盘）。代码与原始数据保留：

```
preliminary_3b_memory_sweep/
├── sweep.py        12 个配置的扫描编排（含子进程隔离 / 超时 killpg / 死法分类）
├── summarize.py    汇总表
└── run.sh
```

原始结果：`small-gpu-vlm-inference/results/preliminary_3b_memory_sweep/`

要复现需先重新下载 3B 模型（`Qwen/Qwen2.5-VL-3B-Instruct` + 其 AWQ 版，约 10.3 GiB），
并且注意上面的结论：**它是"被迫 offload"的场景，不能用来比较引擎**。

## 与当前实验的关系

| | 前置（本目录，3B） | 当前（`vlm_inference_benchmark/`，2B） |
|---|---|---|
| 权重 | 3.17 GiB，**装不下** | 2.74 GiB，**装得下** |
| offload | 被迫 1.0 | **0** |
| 可扫描的维度 | gmu / offload / len / seqs（offload 是噪声） | **gmu / len / seqs / mnbt**（全是真变量） |
| 结论可归因性 | ❌ 只能做工程选型 | ✅ 可归因到引擎与机制 |

→ **这就是"先在小模型上把变量控干净"的价值**：3B 上花了 18 次运行才明白"这个变量消不掉"，
2B 上同样的问题在第一次启动时就自然消失了。