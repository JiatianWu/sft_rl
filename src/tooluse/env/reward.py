"""Reward computation for tau-retail-lite.

The outcome term follows tau-bench: reward is the product of a database-state check and
an output check, so an agent can neither talk its way to a reward without acting nor act
correctly while failing to report back. Everything else is shaping, and every shaping
term is designed so that it cannot be farmed independently of the outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .db import WRITE_ACTIONS, db_hash

# Arguments that must match the oracle for an action to count as progress. Free-text
# arguments (a cancellation reason) are deliberately excluded: they do not affect state.
KEY_ARGS: dict[str, tuple[str, ...]] = {
    "cancel_pending_order": ("order_id",),
    "modify_pending_order_address": ("order_id", "address1", "city", "state", "zip"),
    "return_delivered_order_items": ("order_id", "item_ids"),
    "exchange_delivered_order_items": ("order_id", "item_ids", "new_item_ids"),
    "transfer_to_human": (),
}


@dataclass
class RewardConfig:
    """Weights for the reward components. Exposed so each term can be ablated."""

    w_outcome: float = 1.0
    w_progress: float = 0.3
    w_efficiency: float = 0.05  # penalty per redundant call, only applied on success
    w_violation: float = 0.2  # penalty per illegal write or failed call
    max_efficiency_penalty: float = 0.15
    max_violation_penalty: float = 0.5


@dataclass
class Rollout:
    """What the environment observed during one episode."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    assistant_text: str = ""
    failed_calls: int = 0
    illegal_writes: int = 0


def _norm(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(sorted(_norm(v) for v in value))
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _args_match(name: str, actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key in KEY_ARGS.get(name, ()):
        if _norm(actual.get(key)) != _norm(expected.get(key)):
            return False
    return True


def check_outputs(required: list[list[str]], text: str) -> bool:
    """Every required piece of information must appear, in any accepted rendering."""
    haystack = text.lower()
    return all(any(alt.lower() in haystack for alt in alternatives) for alternatives in required)


def compute_reward(spec, rollout: Rollout, db: dict[str, Any], config: RewardConfig) -> dict[str, float]:
    """Score one episode, returning the total alongside every component for logging."""
    # --- Grounded outcome: state check AND output check, as a product. ---
    r_action = float(db_hash(db) == spec.expected_hash)
    r_output = float(check_outputs(spec.required_outputs, rollout.assistant_text))
    r_outcome = r_action * r_output

    # --- Progress: fraction of oracle actions performed with correct arguments. ---
    # Consumed greedily so repeating one correct call cannot cover for a missing one.
    remaining = list(spec.oracle_actions)
    matched = 0
    for call in rollout.calls:
        for idx, expected in enumerate(remaining):
            if call["name"] == expected["name"] and _args_match(call["name"], call["args"], expected["args"]):
                matched += 1
                remaining.pop(idx)
                break
    if spec.oracle_actions:
        r_progress = matched / len(spec.oracle_actions)
    else:
        # Read-only task: progress means having refrained from writing at all.
        wrote = any(c["name"] in WRITE_ACTIONS for c in rollout.calls)
        r_progress = 0.0 if wrote else 1.0

    # --- Efficiency: only charged on success, so it can never outrank correctness. ---
    seen: set[tuple] = set()
    redundant = 0
    for call in rollout.calls:
        key = (call["name"], repr(_norm(call["args"])))
        if key in seen:
            redundant += 1
        seen.add(key)
    p_efficiency = 0.0
    if r_outcome > 0:
        p_efficiency = -min(config.w_efficiency * redundant, config.max_efficiency_penalty)

    # --- Violations: illegal writes and failed calls are always charged. ---
    p_violation = -min(
        config.w_violation * (rollout.illegal_writes + rollout.failed_calls),
        config.max_violation_penalty,
    )

    total = config.w_outcome * r_outcome + config.w_progress * r_progress + p_efficiency + p_violation

    return {
        "reward": total,
        "r_action": r_action,
        "r_output": r_output,
        "r_outcome": r_outcome,
        "r_progress": r_progress,
        "p_efficiency": p_efficiency,
        "p_violation": p_violation,
        "n_calls": float(len(rollout.calls)),
        "n_redundant": float(redundant),
        "n_failed": float(rollout.failed_calls),
        "n_illegal_writes": float(rollout.illegal_writes),
    }
