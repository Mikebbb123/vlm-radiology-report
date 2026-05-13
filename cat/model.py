"""
VLM 模型加载
"""
import torch
from typing import List

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

from config import TrainingConfig, LoRAConfig


def load_vlm_model(config: TrainingConfig):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        min_pixels=config.image_min_pixels,
        max_pixels=config.image_max_pixels,
    )

    print(f"[Model] Loaded {config.model_name}")
    print(f"[Model] Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model, processor


def create_lora_config(rank: int, target_modules: List[str]) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
