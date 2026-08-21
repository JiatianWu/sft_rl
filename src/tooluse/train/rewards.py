"""Reward functions for GRPO.

Scoring lives in trainer-owned reward functions rather than the environment's
`get_reward` for one reason: the outcome term is the *product* of a state check and an
output check, and the environment never sees the assistant's text. A reward function
receives both the completions and the environment instances, so it can compute the
product; splitting the two halves across two reward sources would sum them instead,
which would let an agent collect half the reward for talking without acting.

The same `compute_reward` used here is used by the evaluation harness, so the training
signal and the reported metric cannot drift apart.
"""

from __future__ import annotations

from typing import Any

from tooluse.env import RewardConfig, compute_reward
from tooluse.eval.harness import TOOL_METHODS, parse_tool_calls, strip_tool_calls

REWARD_CONFIG = RewardConfig()


def _completion_text(completion: Any) -> str:
    """Concatenate the assistant's own words across a multi-turn completion."""
    if isinstance(completion, str):
        return completion
    parts = []
    for message in completion:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(block.get("text", "") for block in content if isinstance(block, dict))
    return " ".join(parts)


def task_reward(completions: list[Any], environments: list[Any], **kwargs: Any) -> list[float]:
    """The grounded reward: state check x output check, plus progress and penalties."""
    log_metric = kwargs.get("log_metric")
    log_extra = kwargs.get("log_extra")

    rewards: list[float] = []
    breakdown: dict[str, list[float]] = {}

    for completion, env in zip(completions, environments):
        text = _completion_text(completion)
        rollout = env._rollout
        rollout.assistant_text = strip_tool_calls(text)
        scores = compute_reward(env.spec, rollout, env._db, REWARD_CONFIG)
        rewards.append(scores["reward"])
        for key, value in scores.items():
            breakdown.setdefault(key, []).append(value)

    if log_metric is not None:
        for key, values in breakdown.items():
            if key != "reward":
                log_metric(f"env/{key}", sum(values) / len(values))
        # Success rate is the number that actually matters; log it explicitly.
        log_metric("env/success_rate", sum(breakdown["r_outcome"]) / len(breakdown["r_outcome"]))
    if log_extra is not None:
        log_extra("family", [env.spec.family for env in environments])
        log_extra("r_outcome", breakdown["r_outcome"])

    return rewards


def protocol_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    """Dense, text-level shaping for tool-call protocol quality.

    This is the densest signal available early in training, when almost every rollout
    fails the task outright and the grounded reward is constant across a group (zero
    advantage). Its weight is annealed toward zero as protocol quality saturates, so it
    stops competing with the real objective once it has done its job.
    """
    trainer_state = kwargs.get("trainer_state")
    log_metric = kwargs.get("log_metric")

    scale = 1.0
    if trainer_state is not None and getattr(trainer_state, "max_steps", 0):
        progress = min(1.0, trainer_state.global_step / trainer_state.max_steps)
        scale = 1.0 - 0.85 * progress  # 1.0 -> 0.15 over the run

    scores: list[float] = []
    for completion in completions:
        text = _completion_text(completion) if not isinstance(completion, str) else completion
        raw = completion if isinstance(completion, str) else _raw_text(completion)
        calls = parse_tool_calls(raw)

        score = 0.0
        if calls:
            score += 0.4  # attempted to act at all
            well_formed = [c for c in calls if not c.get("_malformed")]
            score += 0.3 * (len(well_formed) / len(calls))
            known = [c for c in well_formed if c.get("name") in TOOL_METHODS]
            score += 0.3 * (len(known) / len(calls))
        if strip_tool_calls(raw).strip():
            score = min(1.0, score + 0.1)  # said something to the user, not only tool calls
        scores.append(score * scale)

    if log_metric is not None:
        log_metric("protocol/score", sum(scores) / max(len(scores), 1))
        log_metric("protocol/scale", scale)
    return scores


def _raw_text(completion: Any) -> str:
    """Full assistant output including tool-call markup, for protocol checking."""
    if isinstance(completion, str):
        return completion
    parts = []
    for message in completion:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        for call in message.get("tool_calls") or []:
            function = call.get("function", call)
            name = function.get("name")
            arguments = function.get("arguments")
            parts.append(f'<tool_call>\n{{"name": "{name}", "arguments": {arguments}}}\n</tool_call>')
    return "\n".join(parts)


__all__ = ["REWARD_CONFIG", "protocol_reward", "task_reward"]
