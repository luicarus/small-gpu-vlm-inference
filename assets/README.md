# assets

仓库自带的测试素材。

| 文件 | 说明 |
|---|---|
| `test_image.png` | 实验与冒烟脚本的**默认测试图**（1080×675，真 PNG，731 KB） |

被这些地方引用：

- `engines/{vllm,hf}/vlm_smoke.py` 的 `DEFAULT_IMAGE`
- `experiments/vlm_inference_benchmark/shared/common.py` 的 `DEFAULT_IMAGE`
- `engines/vllm/serve_vlm_chat.sh` 的 HTTP serving 示例

## ⚠️ 换图会改变实验数字

报告与结果文件里的"视觉 token 数"（例如共享前缀的 token 数）
是在**这张图**上、按 `max_pixels=262144` 处理后的实测值。

换一张图（或缩放）后，Qwen2-VL 的 `smart_resize` 会给出不同的 patch 数，
**token 数随之变化，报告里的数字就不再可复现**。换图后必须重跑：

```bash
# 1. 重新生成 workload（图片路径 + 问题集）
python experiments/vlm_inference_benchmark/vllm/exp3_prefix_caching/make_workload.py

# 2. 重跑用到测试图的两个实验
bash experiments/vlm_inference_benchmark/vllm/exp1_memory_boundary/run.sh
bash experiments/vlm_inference_benchmark/vllm/exp3_prefix_caching/run_frames.sh
```

> `exp3_prefix_caching/make_workload.py` 里的 **24 条手写问题是针对图片内容的**，
> 换图后必须同步改写，否则问题与图片不匹配、答案失去意义。

## 版权与来源

`test_image.png` 是**第三方插画作品**，仅作为仓库内的**测试素材**使用，
**不在本仓库的 MIT 许可范围内**（MIT 只覆盖代码与文档）。

若要 fork 或再分发，请替换为自己的图片：

```bash
# 方式一：环境变量临时指定（不动物件）
VLM_IMAGE=/path/to/your.png bash engines/vllm/run_vlm_smoke.sh

# 方式二：替换默认图（记得同步更新 make_workload.py 的问题集）
cp your.png assets/test_image.png
```

引用方式：

```python
from shared.common import DEFAULT_IMAGE   # → 本目录下的 test_image.png
```
