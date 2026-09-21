# exp3_prefix_caching

测量 prefix caching 在「多个请求共享一段长前缀」场景下的收益。

## 怎么跑

```bash
python make_workload.py --questions 50 --frames 16   # 生成 workload（图片路径 + 问题集）

bash run_smoke.sh --num-questions 4    # 冒烟：验证视频输入路径 + 命中情况（首次必须）
bash run_frames.sh                     # 主实验：帧数 8/16/32/40 × 缓存开关，约 15 分钟
NS='5 20 50' bash run.sh               # 附带实验：问题数扫描
bash probe_frames.sh 48 8192           # 帧数可行性探测（KV 池给 max_model_len 的上限）
```

## 文件

| 文件 | 作用 |
|---|---|
| `make_workload.py` | 生成 workload：共享前缀图片 + **24 条针对图片内容的手写问题** + 模板补足到 N |
| `smoke_video.py` | 冒烟：验证 `list[PIL.Image]` 当视频喂进去能跑通，并测出前缀 token 数与命中率 |
| `run_cache.py` | 执行器：逐请求记录命中率与**本进程墙钟** |
| `run_frames.sh` | 主实验入口（帧数扫描） |
| `run.sh` | 附带实验入口（问题数扫描） |

## 设计要点

- **共享前缀 = 同一张图重复 K 帧**（伪视频）。帧完全一致 → 前缀哈希必然一致，变量最干净；
  也复现了「多 QA 共享视频前缀」的形态。
- **对照必须显式关闭**：`--no-enable-prefix-caching`。
  vLLM 0.28 里该特性**默认开启**，不关就是「开 vs 开」，测不出任何差异。
- **`max_num_seqs=1` 串行**：本实验只问「前一个请求的 KV 能否被后一个复用」，
  并发会把变量搅浑（块可能被抢占或淘汰）。
- **热身轮**：`run_frames.sh` 先跑一轮小配置丢弃，避免 Triton 运行时 JIT 的长尾混进数据。

## 输出

`results/vlm_inference_benchmark/vllm/exp3_prefix_caching/`

- `frames<K>_cache{on,off}.jsonl`（8 份）· `cache{on,off}_n<N>.jsonl`（6 份）
- 每条记录：问题、`prompt_tokens`、**`cached_tokens`**、`hit_rate`、`wall_s`、
  以及 vLLM 自报的 `ttft_vllm_s`（**仅供参考，见下**）。
- `logs/<配置>.log`

## 怎么读结果

**主指标是 `wall_s`（本进程墙钟），不是 `ttft_vllm_s`。**

vLLM 的 `RequestOutput.metrics.first_token_latency` 在本配置下**不可用**——
实测出现负值（-2.0 / -1.4 s），因为它的两个时间戳来自不同来源：

```python
# vllm/v1/metrics/stats.py:373
def _time_since(self, start): return self.iteration_timestamp - start
```

同一条链路上的 `prefill_time` / `e2e_latency` 也恒为 0（事件未填充）。

**命中率怎么验证机制**：`cached_tokens` 恒等于前缀 token 数（且总为块大小 16 的整数倍），
不随问题长度变化 → 说明复用是**前缀级**的，每个请求只重算自己的尾巴。

**解码**：统计时**剔除第 0 条请求**（缓存是冷的，且含一次性 JIT 成本），
用后续请求的中位数。仓库里的分析脚本都遵守这条。

## 换测试图之后

`assets/test_image.png` 一换，**token 数就变了**，必须：

1. 重新 `python make_workload.py`；
2. **改写里面 24 条手写问题**（它们是针对图片内容的，不匹配就失去意义）；
3. 重跑 `run_frames.sh`。

详见 `assets/README.md`。

## 相关

- 完整数据与解释：`REPORT.md` §4
- 被推翻的初版结论与排错过程：`REPORT.md` §5.2
