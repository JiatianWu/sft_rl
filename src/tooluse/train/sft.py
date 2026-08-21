"""LoRA supervised fine-tuning on multi-turn tool-use trajectories.

Deliberate deviation from the suggested stack: this uses stock PEFT + `transformers`
rather than Unsloth. Unsloth pins its own `transformers` version, while the GRPO stage
needs `transformers>=5.2` for TRL's `environment_factory`. Keeping one dependency set
removes a version conflict and an image build from a hard time budget; the ~10 minutes
Unsloth would have saved on a 0.6B LoRA run does not pay for that risk.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from tooluse.data.masking import IGNORE_INDEX, build_example
from tooluse.train.lora import build_lora


@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention = [], [], []
        for feature in features:
            pad = width - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad)
            labels.append(feature["labels"] + [IGNORE_INDEX] * pad)
            attention.append([1] * len(feature["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


def load_dataset(path: Path, tokenizer, limit: int | None, max_length: int) -> Dataset:
    rows = [json.loads(line) for line in path.open()]
    if limit:
        rows = rows[:limit]

    examples: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        example = build_example(tokenizer, row["messages"], row["tools"], max_length=max_length)
        if example is None:
            dropped += 1
            continue
        examples.append(example)

    supervised = sum(sum(1 for l in e["labels"] if l != IGNORE_INDEX) for e in examples)
    total = sum(len(e["input_ids"]) for e in examples)
    print(f"[sft] {len(examples)} examples ({dropped} dropped)")
    print(f"[sft] {total / 1e6:.1f}M tokens, {supervised / 1e6:.2f}M supervised ({supervised / max(total, 1):.1%})")
    return Dataset.from_list(examples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data", type=Path, default=Path("data/sft_apigen.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/sft"))
    parser.add_argument("--limit", type=int, default=2500)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=32)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset(args.data, tokenizer, args.limit, args.max_length)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    model = get_peft_model(model, build_lora(args.lora_rank))
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=5,  # transformers 5 removed warmup_ratio
            logging_steps=5,
            save_strategy="no",
            bf16=True,
            report_to=[],
            gradient_checkpointing=True,
            remove_unused_columns=False,
        ),
        train_dataset=dataset,
        data_collator=PadCollator(tokenizer.pad_token_id or tokenizer.eos_token_id),
    )
    result = trainer.train()
    print(f"[sft] final loss: {result.training_loss:.4f}")

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    (args.output / "train_summary.json").write_text(
        json.dumps({"training_loss": result.training_loss, "n_examples": len(dataset)}, indent=2)
    )
    print(f"[sft] saved adapter to {args.output}")


if __name__ == "__main__":
    main()
