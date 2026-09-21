# tools

**跨实验工具**：查看结果、校验文档数字、结构回归。不放实验逻辑，实验在 `experiments/`。

```bash
# 全部命令都在 small-gpu-vlm-inference/ 目录下执行
cd "/mnt/d/Work Places/Python Work Place/Job/InfraStudy/small-gpu-vlm-inference"
```

## 结果查看

| 工具 | 用途 | 用法 |
|---|---|---|
| `peek_results.py` | 逐条明细人工验货（对/错标记、真值对比） | `python tools/peek_results.py results/vlm_inference_benchmark/exp2_engine_comparison/hf_nf4_b1.jsonl --wrong` |

```bash
python tools/peek_results.py <结果.jsonl> [--n 5] [--correct] [--wrong]
```

## 文档与数字校验

> 为什么要这些脚本：周记录与博客是**人工誊写**的，一个数字抄错就会砸掉整篇的可信度。
> 让脚本去和原始 JSONL 对账，比人眼逐行核对可靠。

| 工具 | 校验对象 | 断言 |
|---|---|---|
| `verify_w4_record.py` | `docs/周记录/W4_实操记录_2026-09-15.md` | §3.1/§3.2 每个数字与 JSONL 完全一致，且同一行取自同一批次 |
| `verify_blog_numbers.py` | `docs/blog/知乎_*.md` | 博客引用的每个实测数字与 JSONL 一致 |
| `verify_blog_sources.sh` | vLLM 源码 | 博客引用的源码结论确实存在于本机 venv（offload 只包解码层等） |
| `dump_w4_truth.py` | 原始 JSONL | 打印每个配置的权威值，供人工比对周记录 |
| `list_w4_batches.py` | 原始 JSONL | 按批次列出所有运行，定位"周记录某行取自哪一批" |

```bash
python tools/verify_w4_record.py      # 期望输出：✅ 周记录 §3.1 / §3.2 与 JSONL 完全一致
python tools/verify_blog_numbers.py
bash   tools/verify_blog_sources.sh
python tools/dump_w4_truth.py
python tools/list_w4_batches.py
```

## 结构回归

| 工具 | 用途 |
|---|---|
| `verify_refactor.sh` | 目录重构后确认所有脚本仍互相找得到（不占 GPU）。**改动脚本结构后必须跑一次** |

```bash
bash tools/verify_refactor.sh
```

检查 7 项：W4 扫描器 dry-run、W4 汇总、周记录校验、批次列举、W5 对比分析、W5 结果查看、全部 py 语法编译。

---

## 约定

- 工具只读**不改**数据：发现问题由人来修文档，工具负责指出不一致。
- 工具路径不写死绝对路径（`/mnt/d/...`），一律用 `Path(__file__).resolve().parents[N]` 推算，移动目录不会失效。
- 新增工具请遵循 `<动词>_<对象>.py` 命名并补进本表。
