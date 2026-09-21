import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def main():
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(Path.home() / "models/Qwen2-VL-2B-Instruct-AWQ"),
        quantization="awq",
        # 2B 权重 2.74 GiB 装得下，无需 offload（offload 会让解码走 PCIe，见引擎对比实验）
        cpu_offload_gb=0.0,
        gpu_memory_utilization=0.80,
        max_model_len=256,
        max_num_seqs=1,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"min_pixels": 3136, "max_pixels": 50176},
    )
    prompt = "你好，请做一下自我介绍。"
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=64),
        use_tqdm=False,
    )
    for output in outputs:
        print("PROMPT:", output.prompt)
        print("ANSWER:", output.outputs[0].text)
    print("OFFLINE_TEXT_OK")


if __name__ == "__main__":
    main()
