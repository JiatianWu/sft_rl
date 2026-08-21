"""Multi-turn tool-calling loop and metrics for tau-retail-lite.

This mirrors what TRL's trainer does during RL rollouts (render tools into the chat
template, parse tool calls, execute them, append the result as a tool message) so that
the evaluation numbers describe the same protocol the model is trained under.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from tooluse.env import RetailEnv, RewardConfig, compute_reward
from tooluse.env.retail import SYSTEM_PROMPT
from tooluse.env.tasks import TaskSpec, sample_task

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

TOOL_METHODS = [
    "find_user_id_by_email",
    "get_user_details",
    "list_user_orders",
    "get_order_details",
    "get_product_details",
    "cancel_pending_order",
    "modify_pending_order_address",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
    "transfer_to_human",
]


class Generator(Protocol):
    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str: ...


@dataclass
class EpisodeResult:
    seed: int
    family: str
    difficulty: str
    scores: dict[str, float]
    messages: list[dict[str, Any]] = field(default_factory=list)
    n_turns: int = 0
    stopped_reason: str = ""


def tool_schemas(env: RetailEnv) -> list[dict[str, Any]]:
    """JSON schemas for the environment's tools, in a stable order."""
    from transformers.utils import get_json_schema

    return [get_json_schema(getattr(env, name)) for name in TOOL_METHODS]


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract tool calls from the model's raw output.

    Returns entries with `_malformed` set when the tags are present but the payload does
    not parse, so the caller can distinguish "did not try" from "tried and failed".
    """
    calls = []
    for match in TOOL_CALL_RE.findall(text):
        try:
            payload = json.loads(match)
        except json.JSONDecodeError:
            calls.append({"_malformed": True, "raw": match})
            continue
        name = payload.get("name")
        args = payload.get("arguments", payload.get("parameters", {}))
        if not isinstance(args, dict):
            calls.append({"_malformed": True, "raw": match})
            continue
        calls.append({"name": name, "args": args})
    return calls


def strip_tool_calls(text: str) -> str:
    """The part of the model's turn addressed to the user."""
    cleaned = TOOL_CALL_RE.sub(" ", text)
    cleaned = re.sub(r"<think>.*?</think>", " ", cleaned, flags=re.DOTALL)
    return cleaned.strip()


class Episode:
    """One task as a state machine, so episodes can be advanced in lockstep.

    Batching matters: evaluating three checkpoints over hundreds of tasks and several
    trials is thousands of generations, and issuing them one at a time wastes most of
    the GPU.
    """

    def __init__(self, spec: TaskSpec, reward_config: RewardConfig, max_turns: int = 8) -> None:
        self.spec = spec
        self.reward_config = reward_config
        self.max_turns = max_turns

        self.env = RetailEnv()
        self.env.reset(seed=spec.seed, family=spec.family, difficulty=spec.difficulty)
        self.schemas = tool_schemas(self.env)
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.env.spec.instruction},
        ]

        self._text_parts: list[str] = []
        self._malformed = 0
        self._unknown_tool = 0
        self._turns = 0
        self.done = False
        self.stopped_reason = "max_turns"

    def step(self, raw: str) -> None:
        """Consume one model turn and execute any tool calls it contains."""
        if self.done:
            return
        self._turns += 1
        calls = parse_tool_calls(raw)
        self._text_parts.append(strip_tool_calls(raw))
        self.messages.append({"role": "assistant", "content": raw})

        if not calls:
            self.stopped_reason = "final_answer"
            self.done = True
            return

        for call in calls:
            if call.get("_malformed"):
                self._malformed += 1
                self.messages.append({"role": "tool", "content": "Error: malformed tool call."})
                continue
            method = getattr(self.env, call["name"], None) if call["name"] in TOOL_METHODS else None
            if method is None:
                self._unknown_tool += 1
                self.messages.append({"role": "tool", "content": f"Error: unknown tool {call['name']}."})
                continue
            try:
                result = method(**call["args"])
            except TypeError as exc:
                self.env._rollout.failed_calls += 1
                result = f"Error: bad arguments for {call['name']}: {exc}"
            self.messages.append({"role": "tool", "content": str(result)})

        if self._turns >= self.max_turns:
            self.done = True

    def result(self) -> EpisodeResult:
        rollout = self.env._rollout
        rollout.assistant_text = " ".join(part for part in self._text_parts if part)
        # Malformed calls and hallucinated tool names never reach the environment, so
        # fold them into the violation count here.
        rollout.failed_calls += self._malformed + self._unknown_tool

        scores = compute_reward(self.env.spec, rollout, self.env._db, self.reward_config)
        scores["n_malformed"] = float(self._malformed)
        scores["n_unknown_tool"] = float(self._unknown_tool)
        scores["called_any_tool"] = float(bool(rollout.calls))

        return EpisodeResult(
            seed=self.spec.seed,
            family=self.spec.family,
            difficulty=self.spec.difficulty,
            scores=scores,
            messages=self.messages,
            n_turns=self._turns,
            stopped_reason=self.stopped_reason,
        )


def run_episode(
    generate: Generator,
    spec: TaskSpec,
    reward_config: RewardConfig,
    max_turns: int = 8,
) -> EpisodeResult:
    """Run one task end to end with a synchronous generator."""
    episode = Episode(spec, reward_config, max_turns)
    while not episode.done:
        episode.step(generate(episode.messages, episode.schemas))
    return episode.result()


def pass_hat_k(successes: int, trials: int, k: int) -> float:
    """tau-bench's pass^k: the chance that a random size-k subset of trials all pass."""
    if k > trials:
        return float("nan")
    if successes < k:
        return 0.0
    return math.comb(successes, k) / math.comb(trials, k)


def aggregate(results: list[EpisodeResult], n_trials: int) -> dict[str, Any]:
    """Summarise per-task trials into pass^k plus the decomposed reward components."""
    by_task: dict[tuple[str, int, str], list[EpisodeResult]] = defaultdict(list)
    for result in results:
        by_task[(result.family, result.seed, result.difficulty)].append(result)

    summary: dict[str, Any] = {}
    for k in range(1, n_trials + 1):
        values = [
            pass_hat_k(sum(r.scores["r_outcome"] > 0 for r in trials), len(trials), k)
            for trials in by_task.values()
            if len(trials) >= k
        ]
        summary[f"pass^{k}"] = sum(values) / len(values) if values else float("nan")

    numeric_keys = [
        "r_action",
        "r_output",
        "r_outcome",
        "r_progress",
        "n_calls",
        "n_redundant",
        "n_failed",
        "n_illegal_writes",
        "n_malformed",
        "n_unknown_tool",
        "called_any_tool",
        "reward",
    ]
    for key in numeric_keys:
        values = [r.scores.get(key, 0.0) for r in results]
        summary[key] = sum(values) / len(values) if values else 0.0

    per_family: dict[str, float] = {}
    family_groups: dict[str, list[EpisodeResult]] = defaultdict(list)
    for result in results:
        family_groups[result.family].append(result)
    for family, group in family_groups.items():
        per_family[family] = sum(r.scores["r_outcome"] for r in group) / len(group)
    summary["per_family_success"] = per_family
    summary["n_episodes"] = len(results)
    summary["n_tasks"] = len(by_task)
    return summary


def build_task_split(
    seeds: range | list[int],
    families: list[str],
    difficulty: str = "easy",
) -> list[TaskSpec]:
    return [sample_task(seed, family, difficulty) for seed in seeds for family in families]


__all__ = [
    "EpisodeResult",
    "aggregate",
    "build_task_split",
    "parse_tool_calls",
    "pass_hat_k",
    "run_episode",
    "strip_tool_calls",
    "tool_schemas",
]
