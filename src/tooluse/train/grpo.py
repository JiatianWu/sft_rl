"""Online RL (GRPO) against tau-retail-lite.

TRL's `environment_factory` owns the multi-turn rollout: it builds one `RetailEnv` per
rollout, exposes its public methods as tools, runs the tool loop, and appends results to
the conversation. That leaves this file responsible only for the split, the reward
weighting and the optimisation settings.

Note: `environment_factory` is flagged experimental in TRL, so the API may move.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from tooluse.data.masking import CHAT_TEMPLATE_KWARGS
from tooluse.env.retail import RetailEnv, build_prompt
from tooluse.env.splits import RL_TRAIN_FAMILIES, TRAIN_SEEDS
from tooluse.train.rewards import protocol_reward, task_reward


def build_train_dataset(n_tasks: int, seed: int = 0) -> Dataset:
    """Sample RL tasks. Columns other than `prompt` are forwarded to `RetailEnv.reset`."""
    rng = random.Random(seed)
    seeds = rng.sample(list(TRAIN_SEEDS), min(n_tasks, len(TRAIN_SEEDS)))
    rows = []
    for task_seed in seeds:
        rows.append(
            {
                "prompt": build_prompt(),
                "seed": task_seed,
                "family": rng.choice(RL_TRAIN_FAMILIES),
                "difficulty": "easy",
            }
        )
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", type=Path, default=None, help="SFT adapter to continue training")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/grpo"))
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--protocol-weight", type=float, default=0.3)
    parser.add_argument("--no-vllm", action="store_true")
    parser.add_argument("--vllm-memory", type=float, default=0.25)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)

    if args.adapter is None:
        # RL-only ablation: same adapter capacity, no SFT initialisation. This is what
        # isolates SFT's contribution from RL's.
        from peft import get_peft_model

        from tooluse.train.lora import build_lora

        model = get_peft_model(model, build_lora())
        print("[grpo] fresh LoRA (no SFT initialisation)")
    else:
        from peft import PeftModel

        # Continue training the SFT adapter rather than starting a fresh one, so RL
        # refines the policy SFT produced instead of competing with it.
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=True)
        print(f"[grpo] continuing from SFT adapter {args.adapter}")

    dataset = build_train_dataset(n_tasks=args.steps * args.batch_size * args.grad_accum)
    print(f"[grpo] {len(dataset)} tasks over families {RL_TRAIN_FAMILIES}")

    config = GRPOConfig(
        output_dir=str(args.output),
        max_steps=args.steps,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        bf16=True,
        max_completion_length=512,
        max_tool_calling_iterations=args.max_turns,
        chat_template_kwargs=CHAT_TEMPLATE_KWARGS,
        reward_weights=[1.0, args.protocol_weight],
        use_vllm=not args.no_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_memory,
        vllm_max_model_length=16384,
        temperature=1.0,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        log_completions=True,
        num_completions_to_print=2,
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=[task_reward, protocol_reward],
        environment_factory=RetailEnv,
    )
    trainer.train()

    args.output.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"[grpo] saved adapter to {args.output}")


if __name__ == "__main__":
    main()
