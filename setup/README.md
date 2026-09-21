# setup

**环境安装与体检。** 一次性的安装脚本 + 随时可跑的诊断脚本，不放实验逻辑。

## 安装

| 脚本 | 装什么 | 备注 |
|---|---|---|
| `install_vllm.sh` | vLLM + Python venv | 需 root（`apt` 装 python3-venv），创建 `~/venvs/vllm` |
| `install_nvcc.sh` | CUDA 工具链 | 需要 nvcc 时用 |
| `install_bitsandbytes.sh` | bitsandbytes（NF4 量化依赖） | 用 `--no-deps` 保护 vLLM 依赖树；装完自动回归验证 |
| `install_accelerate.sh` | accelerate | HF 侧 `device_map="auto"` 需要；同样 `--no-deps` |
| `download_qwen2vl_2b.sh` | `Qwen2-VL-2B-Instruct` + 其 AWQ 版（约 7 GiB） | 走 `hf-mirror.com` |

```bash
bash setup/install_vllm.sh
bash setup/install_bitsandbytes.sh
bash setup/install_accelerate.sh
bash setup/download_qwen2vl_2b.sh
```

> **为什么装库要 `--no-deps`**：vLLM 对 torch / transformers / numpy 的版本很敏感，
> 让 pip 自动解析依赖可能把它们升级掉，直接搞坏已跑通的引擎环境。
> 安装脚本会在结束后 import vllm + transformers + torch 做回归验证。

## 体检

| 脚本 | 查什么 |
|---|---|
| `gpu_check.sh` | 整卡占用 / 温度 / 功率 + 是否有 `EngineCore` 孤儿进程残留 |
| `probe_gpu_real.py` | 跑真实计算，测出**实际可分配的显存上限**（不只看读数） |

```bash
bash setup/gpu_check.sh
python setup/probe_gpu_real.py
```

**判读要点**：

- `memory.used` 应为 0~800 MiB（Windows 桌面会占用，浮动属正常）；接近总量说明有残留进程。
- `utilization.gpu` 可能是**假读数**——曾见到恒定 100% / 84°C 而没有任何进程在用。
  要确认 GPU 真能用，**得跑一次真实计算**（这正是 `probe_gpu_real.py` 的用途）。
- 在 WDDM 下，CUDA 侧可分配的量可能**超过**物理显存（会溢出到系统内存），
  所以"能分配"不等于"干净"，判据要结合 `memory.used` 与进程表一起看。

## 与其他目录的关系

- 结构回归自检：`tools/verify_refactor.sh`
- 引擎冒烟：`engines/{vllm,hf}/`
- 实验：`experiments/`，结果落 `results/`
