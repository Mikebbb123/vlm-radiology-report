"""
Visual Feature Fusion (VFF) for Qwen2-VL

核心: 把 DenseNet 的 1024-d penultimate feature 通过 gated cross-attention
注入到 Qwen2-VL 的 merger 输出 (vision tokens), 让 VLM 的语言侧能吃到
医学专用的视觉特征.

架构:
    merged_tokens [total_N, 3584]  (Qwen merger 输出, 展平的所有 patch)
        │
        ├─ 按 grid_thw 切分成 per-image
        │
        ├─ medical_feat [B, 1024] → Linear → [B, 3584]
        │
        ├─ 对每个 sample 做 cross-attention:
        │    Q = image_tokens [N_i, 3584]
        │    K,V = medical_feat [1, 3584]
        │    → attn_out [N_i, 3584]
        │
        ├─ Gated residual:
        │    output = image_tokens + tanh(gate) * attn_out
        │
        └─ 拼回 [total_N, 3584]

关键设计:
    - gate 初始化为 0, tanh(0)=0, 训练开始时严格等同于 baseline (下界保护)
    - Cross-attention 让每个 vision token 能独立决定从 medical feat 吸多少信息
"""
import torch
import torch.nn as nn
from typing import List


class VisualFeatureFusion(nn.Module):
    """
    把 medical feature 融合进 Qwen2-VL 的 merged vision tokens.

    输入:
        merged_tokens: [total_merged_tokens, vlm_dim] (Qwen merger 的 pooler_output)
        token_counts: List[int], 每个样本在 merger 输出里占多少个 token
        medical_feats: [B, medical_dim] 每个样本一个 DenseNet feature
                       其中 B == len(token_counts)
    输出:
        fused_tokens: [total_merged_tokens, vlm_dim] (和输入同 shape)
    """
    def __init__(self,
                 vlm_dim: int = 3584,
                 medical_dim: int = 1024,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 init_gate: float = 0.0):
        super().__init__()
        self.vlm_dim = vlm_dim
        self.medical_dim = medical_dim

        # 把 medical feature 投影到 VLM 维度
        self.med_proj = nn.Linear(medical_dim, vlm_dim)

        # Cross-attention: vision tokens (Q) × medical (K, V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=vlm_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # LayerNorm 稳定训练
        self.norm_q = nn.LayerNorm(vlm_dim)
        self.norm_kv = nn.LayerNorm(vlm_dim)

        # Gated residual: tanh(gate) 初始为 0 → 严格等同于 baseline
        self.gate_raw = nn.Parameter(torch.full((1,), init_gate))

    @property
    def current_gate(self) -> float:
        return float(torch.tanh(self.gate_raw).item())

    def forward(self,
                merged_tokens: torch.Tensor,
                token_counts: List[int],
                medical_feats: torch.Tensor) -> torch.Tensor:
        """
        merged_tokens: [total_N, vlm_dim]
        token_counts: List[int], 原始(未经 beam 扩展的)sum
        medical_feats: [B, medical_dim], B = len(token_counts)

        自动处理 beam search: 如果 merged_tokens.shape[0] 是 sum(token_counts) 的
        整数倍 (比如 × num_beams), 自动把 token_counts 和 medical_feats 按倍数扩展.
        """
        expected_sum = sum(token_counts)
        actual_N = merged_tokens.shape[0]

        if actual_N != expected_sum:
            if actual_N % expected_sum == 0:
                mult = actual_N // expected_sum
                # Beam search: 整个 token sequence 被复制了 mult 次
                token_counts = token_counts * mult  # list repeat
                medical_feats = medical_feats.repeat_interleave(mult, dim=0)
            else:
                raise AssertionError(
                    f"sum(token_counts)={expected_sum}, "
                    f"merged_tokens.shape[0]={actual_N}, "
                    f"not an integer multiple (beam mismatch?)"
                )

        assert len(token_counts) == medical_feats.shape[0]

        med_proj = self.med_proj(medical_feats.to(merged_tokens.dtype))
        med_proj = self.norm_kv(med_proj)
        med_proj = med_proj.unsqueeze(1)

        outputs = []
        offset = 0
        for i, count in enumerate(token_counts):
            q = merged_tokens[offset:offset + count]
            q_norm = self.norm_q(q).unsqueeze(0)
            kv = med_proj[i:i + 1]

            attn_out, _ = self.cross_attn(q_norm, kv, kv)
            attn_out = attn_out.squeeze(0)

            gate = torch.tanh(self.gate_raw)
            fused = q + gate * attn_out
            outputs.append(fused)
            offset += count

        return torch.cat(outputs, dim=0)


def compute_token_counts_from_grid_thw(grid_thw: torch.Tensor,
                                         spatial_merge_size: int = 2) -> List[int]:
    """
    根据 image_grid_thw 计算每张图在 merger 输出里占的 token 数.

    grid_thw: [num_images, 3], 每行是 [t, h, w] (merger 前的 patch 数)
    merger 之后: t * (h / merge) * (w / merge) 个 token

    返回 List[int], 长度 = num_images
    """
    counts = []
    for row in grid_thw:
        t, h, w = int(row[0]), int(row[1]), int(row[2])
        n = t * (h // spatial_merge_size) * (w // spatial_merge_size)
        counts.append(n)
    return counts


def _find_merger(model):
    """
    递归查找 Qwen2-VL 的 merger (PatchMerger) 模块.
    兼容以下包装情况:
      - 原始 Qwen2VLForConditionalGeneration:
            model.model.visual.merger
      - PEFT 包装后 (PeftModel / LoraModel):
            model.base_model.model.model.visual.merger
      - 其他包装

    策略: 用 named_modules 搜索, 找名字以 '.visual.merger' 结尾的那个.
    """
    candidates = []
    for name, mod in model.named_modules():
        if name.endswith("visual.merger") or name == "visual.merger":
            candidates.append((name, mod))

    if len(candidates) == 0:
        # 穷举打印 visual 相关模块, 方便 debug
        found = [n for n, _ in model.named_modules() if "visual" in n and "." not in n.replace("visual", "")[:5]]
        raise AttributeError(
            f"Cannot find visual.merger in model. "
            f"Model type: {type(model).__name__}. "
            f"visual-related top modules: {found[:10]}"
        )
    if len(candidates) > 1:
        print(f"[VFF] Warning: found {len(candidates)} merger candidates, using first: {candidates[0][0]}")

    print(f"[VFF] Found merger at: {candidates[0][0]}")
    return candidates[0][1]


def install_vff_hook(model, vff_adapter: VisualFeatureFusion,
                      spatial_merge_size: int = 2):
    """
    在 Qwen2-VL 的 merger 上注册 forward hook, 拦截 output 注入 medical feature.

    model 需要在 forward 前设置两个属性:
        model._vff_medical_feats: [B, medical_dim] tensor
        model._vff_grid_thw: [num_images, 3] tensor (来自 batch 的 image_grid_thw)

    返回 hook handle, 调用 .remove() 可以卸载.
    """
    def hook(module, inputs, output):
        medical_feats = getattr(model, "_vff_medical_feats", None)
        grid_thw = getattr(model, "_vff_grid_thw", None)

        if medical_feats is None or grid_thw is None:
            return output

        per_image_counts = compute_token_counts_from_grid_thw(
            grid_thw, spatial_merge_size=spatial_merge_size
        )

        num_images = len(per_image_counts)
        B = medical_feats.shape[0]

        if num_images == 2 * B:
            per_sample_counts = []
            for i in range(B):
                per_sample_counts.append(per_image_counts[2*i] + per_image_counts[2*i + 1])
        elif num_images == B:
            per_sample_counts = per_image_counts
        else:
            raise ValueError(
                f"num_images ({num_images}) must be B ({B}) or 2*B ({2*B})"
            )

        fused = vff_adapter(output, per_sample_counts, medical_feats)
        return fused

    merger = _find_merger(model)
    handle = merger.register_forward_hook(hook)
    return handle
