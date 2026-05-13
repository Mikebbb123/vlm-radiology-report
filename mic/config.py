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
    """
    默认使用 nucleus sampling (Round 2 配置)。
    - 原 beam=4 / length_penalty=1.2 / rep_penalty=1.2 / no_repeat_ngram=3 已确认会诱发模板化输出
    - 消融实验可通过 overrides 切回 beam 模式做对比
    """
    # --- 采样策略 (默认启用) ---
    do_sample: bool = True
    top_p: float = 0.9
    temperature: float = 0.7

    # --- 重复控制 (放松) ---
    repetition_penalty: float = 1.05   # 1.2 过强，会把模型逼到拼接 MIMIC 残句
    no_repeat_ngram_size: int = 0      # 放射学报告天然重复，禁止 3-gram 有害

    # --- 长度 ---
    max_new_tokens: int = 150
    min_new_tokens: int = 20

    # --- Beam search 参数 (保留给消融实验用，默认不启用) ---
    num_beams: int = 1
    length_penalty: float = 1.0
    early_stopping: bool = False


@dataclass
class DataConfig:
    annotation_file: str = "/content/drive/MyDrive/annotation.json"
    images_dir: str = "/content/drive/MyDrive/iu_xray/images"