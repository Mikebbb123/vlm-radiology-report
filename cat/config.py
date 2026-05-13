"""
Disease-Aware VLM Report Generation
Qwen2-VL-7B + LoRA + Disease-Guided Prompting
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    max_seq_length: int = 512
    use_4bit: bool = False

    image_min_pixels: int = 200704
    image_max_pixels: int = 200704

    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    bf16: bool = True

    discourse_epochs: int = 8
    discourse_batch_size: int = 1
    discourse_grad_accum: int = 16
    discourse_lora_r: int = 16

    output_dir: str = "/content/drive/MyDrive/medk_lora_r2gen"
    logging_steps: int = 10


@dataclass
class GenerationConfig:
    num_beams: int = 4
    length_penalty: float = 1.2
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3
    early_stopping: bool = True
    max_new_tokens: int = 128
    min_new_tokens: int = 15
    do_sample: bool = False


@dataclass
class DataConfig:
    annotation_file: str = "/content/drive/MyDrive/annotation.json"
    images_dir: str = "/content/drive/MyDrive/iu_xray/images"
