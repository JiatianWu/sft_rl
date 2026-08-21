"""End-to-end checks of the tool-calling loop, using scripted generators instead of a GPU.

Running an oracle *through the parser and the loop* catches a whole class of bugs that
testing the environment alone does not: bad tool-call formatting, argument marshalling,
and the loop's stopping behaviour.
"""

from __future__ import annotations

import json

import pytest

from tooluse.env import FAMILIES, RewardConfig, sample_task
from tooluse.eval.harness import aggregate, parse_tool_calls, pass_hat_k, run_episode

CONFIG = RewardConfig()


def _oracle_generator(spec):
    """Emit the oracle's tool calls one per turn, then report the required information."""
    queue = list(spec.oracle_actions)

    def generate(messages, tools):
        if queue:
            action = queue.pop(0)
            args = dict(action["args"])
            if action["name"] == "transfer_to_human":
                args = {"summary": "policy does not allow this"}
            return f'<tool_call>\n{json.dumps({"name": action["name"], "arguments": args})}\n</tool_call>'
        return "All done. " + " ".join(alts[0] for alts in spec.required_outputs)

    return generate


def _silent_generator(messages, tools):
    return "I'm sorry, I can't help with that."


def _malformed_generator(messages, tools):
    return "<tool_call>\n{not valid json,,}\n</tool_call>"


def _hallucinating_generator(messages, tools):
    return '<tool_call>\n{"name": "refund_everything", "arguments": {}}\n</tool_call>'


@pytest.mark.parametrize("family", FAMILIES)
def test_oracle_solves_through_the_loop(family: str) -> None:
    for seed in range(15):
        spec = sample_task(seed, family, "easy")
        result = run_episode(_oracle_generator(spec), spec, CONFIG)
        assert result.scores["r_outcome"] == 1.0, f"{family}/{seed} failed: {result.scores}"
        assert result.scores["n_malformed"] == 0.0
        assert result.scores["n_unknown_tool"] == 0.0
        assert result.stopped_reason == "final_answer"


@pytest.mark.parametrize("family", FAMILIES)
def test_silent_model_fails(family: str) -> None:
    spec = sample_task(0, family, "easy")
    result = run_episode(_silent_generator, spec, CONFIG)
    assert result.scores["r_outcome"] == 0.0
    assert result.scores["called_any_tool"] == 0.0


def test_refusal_requires_escalating_not_just_apologising() -> None:
    """Regression: a model that only says "I can't help" must not score on a refusal task.

    Before `transfer_to_human` had a state footprint, doing nothing produced the correct
    final database and the task had no required outputs, so passivity scored a perfect 1.0.
    """
    spec = sample_task(0, "refuse_invalid", "easy")
    passive = run_episode(_silent_generator, spec, CONFIG)
    assert passive.scores["r_outcome"] == 0.0
    assert passive.scores["r_action"] == 0.0

    escalating = run_episode(_oracle_generator(spec), spec, CONFIG)
    assert escalating.scores["r_outcome"] == 1.0


def test_malformed_calls_are_counted_and_penalised() -> None:
    spec = sample_task(0, "cancel_order", "easy")
    result = run_episode(_malformed_generator, spec, CONFIG, max_turns=3)
    assert result.scores["n_malformed"] == 3.0
    assert result.scores["p_violation"] < 0.0
    assert result.stopped_reason == "max_turns"


def test_hallucinated_tool_is_counted() -> None:
    spec = sample_task(0, "cancel_order", "easy")
    result = run_episode(_hallucinating_generator, spec, CONFIG, max_turns=2)
    assert result.scores["n_unknown_tool"] == 2.0
    assert result.scores["r_outcome"] == 0.0


def test_parse_tool_calls_handles_variants() -> None:
    good = '<tool_call>\n{"name": "get_order_details", "arguments": {"order_id": "#W1"}}\n</tool_call>'
    assert parse_tool_calls(good) == [{"name": "get_order_details", "args": {"order_id": "#W1"}}]

    assert parse_tool_calls("just talking") == []
    assert parse_tool_calls("<tool_call>{oops}</tool_call>")[0]["_malformed"] is True

    two = good + "\n" + good
    assert len(parse_tool_calls(two)) == 2


def test_pass_hat_k_matches_definition() -> None:
    assert pass_hat_k(4, 4, 1) == 1.0
    assert pass_hat_k(4, 4, 4) == 1.0
    assert pass_hat_k(0, 4, 1) == 0.0
    assert pass_hat_k(2, 4, 1) == 0.5
    # 2 of 4 passing: only one of the six possible pairs is all-pass.
    assert pass_hat_k(2, 4, 2) == pytest.approx(1 / 6)


def test_aggregate_reports_passk_and_families() -> None:
    results = []
    for family in ["cancel_order", "return_items"]:
        for seed in range(3):
            spec = sample_task(seed, family, "easy")
            for _ in range(2):
                results.append(run_episode(_oracle_generator(spec), spec, CONFIG))
    summary = aggregate(results, n_trials=2)
    assert summary["pass^1"] == 1.0
    assert summary["pass^2"] == 1.0
    assert summary["n_tasks"] == 6
    assert set(summary["per_family_success"]) == {"cancel_order", "return_items"}
