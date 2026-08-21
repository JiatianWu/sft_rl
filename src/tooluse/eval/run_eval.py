"""Evaluate a checkpoint on the held-out tau-retail-lite split.

Every checkpoint is evaluated with identical decoding, identical prompts and identical
chat-template settings; otherwise the pre/post-SFT/post-RL comparison measures the
harness rather than the model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from tooluse.data.masking import CHAT_TEMPLATE_KWARGS
from tooluse.env import RewardConfig
from tooluse.env.splits import EVAL_FAMILIES, TEST_SEEDS
from tooluse.env.tasks import sample_task
from tooluse.eval.harness import Episode, aggregate


def build_specs(n_seeds: int, families: list[str], difficulty: str) -> list[Any]:
    seeds = list(TEST_SEEDS)[:n_seeds]
    return [sample_task(seed, family, difficulty) for seed in seeds for family in families]


def run(
    model: str,
    adapter: str | None,
    specs: list[Any],
    trials: int,
    max_turns: int,
    temperature: float,
    max_tokens: int,
    seed: int,
) -> list:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = LLM(
        model=model,
        enable_lora=adapter is not None,
        max_lora_rank=64,
        max_model_len=16384,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
    )
    lora_request = LoRARequest("adapter", 1, adapter) if adapter else None
    sampling = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens, seed=seed)

    reward_config = RewardConfig()
    episodes = [Episode(spec, reward_config, max_turns) for spec in specs for _ in range(trials)]

    turn = 0
    while True:
        active = [e for e in episodes if not e.done]
        if not active:
            break
        turn += 1
        prompts = [
            tokenizer.apply_chat_template(
                e.messages,
                tools=e.schemas,
                tokenize=False,
                add_generation_prompt=True,
                **CHAT_TEMPLATE_KWARGS,
            )
            for e in active
        ]
        outputs = llm.generate(prompts, sampling, lora_request=lora_request, use_tqdm=False)
        for episode, output in zip(active, outputs):
            episode.step(output.outputs[0].text)
        print(f"[eval] turn {turn}: advanced {len(active)} episodes, {sum(e.done for e in episodes)} finished")

    return [e.result() for e in episodes]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--tag", required=True, help="checkpoint label, e.g. base / sft / grpo")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    specs = build_specs(args.n_seeds, EVAL_FAMILIES, args.difficulty)
    print(f"[eval] {args.tag}: {len(specs)} tasks x {args.trials} trials")

    started = time.time()
    results = run(
        args.model,
        args.adapter,
        specs,
        args.trials,
        args.max_turns,
        args.temperature,
        args.max_tokens,
        args.seed,
    )
    elapsed = time.time() - started

    summary = aggregate(results, args.trials)
    summary["tag"] = args.tag
    summary["adapter"] = args.adapter
    summary["trials"] = args.trials
    summary["temperature"] = args.temperature
    summary["elapsed_s"] = round(elapsed, 1)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    with (args.out / f"{args.tag}_episodes.jsonl").open("w") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    {
                        "seed": result.seed,
                        "family": result.family,
                        "scores": result.scores,
                        "n_turns": result.n_turns,
                        "stopped_reason": result.stopped_reason,
                    }
                )
                + "\n"
            )
    # A handful of full transcripts, because aggregate numbers never explain a failure.
    with (args.out / f"{args.tag}_transcripts.json").open("w") as handle:
        json.dump([{"family": r.family, "seed": r.seed, "messages": r.messages} for r in results[:12]], handle, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "per_family_success"}, indent=2))
    print("per-family success:", json.dumps(summary["per_family_success"], indent=2))


if __name__ == "__main__":
    main()
