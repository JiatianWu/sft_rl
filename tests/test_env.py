"""Environment sanity checks.

The load-bearing test is `test_oracle_gets_full_reward`: if the oracle action list does
not score 1.0 through the real tool interface, the task is unsolvable and any RL result
on it is meaningless.
"""

from __future__ import annotations

import pytest

from tooluse.env import FAMILIES, RetailEnv, RewardConfig, compute_reward, sample_task
from tooluse.env.db import WRITE_ACTIONS, db_hash

SEEDS = list(range(40))
CONFIG = RewardConfig()


def _run_oracle(env: RetailEnv) -> None:
    """Execute the task's oracle actions through the environment's public tools."""
    for action in env.spec.oracle_actions:
        args = dict(action["args"])
        if action["name"] == "transfer_to_human":
            args = {"summary": "policy does not allow this request"}
        getattr(env, action["name"])(**args)


def _oracle_text(env: RetailEnv) -> str:
    """The minimal assistant text that satisfies the output check."""
    return " ".join(alternatives[0] for alternatives in env.spec.required_outputs)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("difficulty", ["easy", "hard"])
def test_tasks_are_well_formed(family: str, difficulty: str) -> None:
    for seed in SEEDS:
        spec = sample_task(seed, family, difficulty)
        assert spec.instruction.strip()
        assert spec.expected_hash
        # A task must be reachable: every oracle write must be legal from the initial state.
        for action in spec.oracle_writes:
            assert action["name"] in WRITE_ACTIONS


@pytest.mark.parametrize("family", FAMILIES)
def test_oracle_gets_full_reward(family: str) -> None:
    for seed in SEEDS:
        env = RetailEnv()
        env.reset(seed=seed, family=family, difficulty="easy")
        _run_oracle(env)
        rollout = env._rollout
        rollout.assistant_text = _oracle_text(env)
        scores = compute_reward(env.spec, rollout, env._db, CONFIG)

        assert scores["r_action"] == 1.0, f"{family}/{seed}: state mismatch"
        assert scores["r_output"] == 1.0, f"{family}/{seed}: output check failed"
        assert scores["r_outcome"] == 1.0
        assert scores["r_progress"] == 1.0
        assert scores["p_violation"] == 0.0, f"{family}/{seed}: oracle triggered a violation"
        assert scores["reward"] == pytest.approx(CONFIG.w_outcome + CONFIG.w_progress)


@pytest.mark.parametrize("family", FAMILIES)
def test_doing_nothing_scores_below_oracle(family: str) -> None:
    for seed in SEEDS[:10]:
        env = RetailEnv()
        env.reset(seed=seed, family=family, difficulty="easy")
        rollout = env._rollout
        rollout.assistant_text = "I am not sure how to help with that."
        scores = compute_reward(env.spec, rollout, env._db, CONFIG)
        assert scores["reward"] < CONFIG.w_outcome + CONFIG.w_progress


def test_reward_requires_both_action_and_output() -> None:
    """Correct state with no report, and a correct report with no action, must both fail."""
    env = RetailEnv()
    env.reset(seed=1, family="cancel_order", difficulty="easy")
    _run_oracle(env)
    env._rollout.assistant_text = "Done."  # acted, but never reported the refund
    acted_only = compute_reward(env.spec, env._rollout, env._db, CONFIG)
    assert acted_only["r_action"] == 1.0
    assert acted_only["r_output"] == 0.0
    assert acted_only["r_outcome"] == 0.0

    env2 = RetailEnv()
    env2.reset(seed=1, family="cancel_order", difficulty="easy")
    env2._rollout.assistant_text = _oracle_text(env2)  # reported, but never acted
    talked_only = compute_reward(env2.spec, env2._rollout, env2._db, CONFIG)
    assert talked_only["r_output"] == 1.0
    assert talked_only["r_action"] == 0.0
    assert talked_only["r_outcome"] == 0.0


def test_refuse_invalid_punishes_writing() -> None:
    """The refusal family must make 'write anyway' strictly worse than refusing."""
    env = RetailEnv()
    env.reset(seed=3, family="refuse_invalid", difficulty="easy")
    order_id = env.spec.instruction.split("order ")[1].split(",")[0].strip()
    env.cancel_pending_order(order_id=order_id, reason="user asked")
    env._rollout.assistant_text = "Cancelled."
    scores = compute_reward(env.spec, env._rollout, env._db, CONFIG)
    assert scores["n_illegal_writes"] >= 1.0
    assert scores["p_violation"] < 0.0


def test_read_only_task_penalises_writes() -> None:
    env = RetailEnv()
    env.reset(seed=5, family="lookup_and_report", difficulty="easy")
    assert env.spec.oracle_actions == []
    env._rollout.assistant_text = _oracle_text(env)
    clean = compute_reward(env.spec, env._rollout, env._db, CONFIG)
    assert clean["r_progress"] == 1.0
    assert clean["r_outcome"] == 1.0


def test_reset_fully_reinitialises_pooled_instance() -> None:
    """TRL reuses instances across batches, so a stale rollout would corrupt scoring."""
    env = RetailEnv()
    env.reset(seed=7, family="cancel_order", difficulty="easy")
    _run_oracle(env)
    assert env._rollout.calls
    dirty_hash = db_hash(env._db)

    env.reset(seed=7, family="cancel_order", difficulty="easy")
    assert env._rollout.calls == []
    assert env._rollout.illegal_writes == 0
    assert db_hash(env._db) != dirty_hash


def test_tool_schemas_render() -> None:
    """Every tool must produce a valid JSON schema, or the model never sees it."""
    from transformers.utils import get_json_schema

    env = RetailEnv()
    env.reset(seed=0, family="cancel_order")
    tool_names = [
        name
        for name in dir(env)
        if not name.startswith("_") and name not in ("reset", "get_reward", "spec") and callable(getattr(env, name))
    ]
    assert len(tool_names) == 10, tool_names
    for name in tool_names:
        schema = get_json_schema(getattr(env, name))
        function = schema["function"]
        assert function["description"], name
        for param in function["parameters"]["properties"].values():
            assert param.get("description"), name
