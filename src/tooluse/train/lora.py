"""The single LoRA configuration used by every training stage.

SFT and the RL-only ablation must differ in exactly one thing: whether the adapter starts
from SFT weights or from scratch. Defining the config twice would let capacity or target
modules drift silently and turn the ablation into a comparison of two unrelated runs.
"""

from __future__ import annotations

from peft import LoraConfig

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_lora(rank: int = 32) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )


__all__ = ["TARGET_MODULES", "build_lora"]
