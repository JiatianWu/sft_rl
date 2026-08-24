"""End-to-end checks of the tool-calling loop, using scripted generators instead of a GPU.

Running an oracle *through the parser and the loop* catches a whole class of bugs that
testing the environment alone does not: bad tool-call formatting, argument marshalling,
and the loop's stopping behaviour.
"""

from __future__ import annotations

import json

import pytest

from tooluse.env import FAMILIES, RewardConfig, sample_task
from tooluse.env.splits import EVAL_FAMILIES, TEST_SEEDS
from tooluse.env.tasks import ABSTAIN_FAMILY
from tooluse.eval.harness import aggregate, parse_tool_calls, pass_hat_k, run_episode
from tooluse.eval.run_eval import build_specs

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


def _one_read_then_decline(messages, tools):
    """A lookup, then a perfectly worded refusal — the behaviour RL currently rewards."""
    if not any(m.get("role") == "tool" for m in messages):
        return '<tool_call>\n{"name": "list_user_orders", "arguments": {"user_id": "U1"}}\n</tool_call>'
    return "I'm sorry, I can't help with that."


def _escalating_generator(messages, tools):
    if not any(m.get("role") == "tool" for m in messages):
        return '<tool_call>\n{"name": "transfer_to_human", "arguments": {"summary": "out of scope"}}\n</tool_call>'
    return "I can't help with that, so I've passed you to a colleague."


def _cheerful_generator(messages, tools):
    return "Sure thing, happy to help with that!"


def test_abstaining_scores_only_when_nothing_is_called() -> None:
    """The one behaviour `tau-retail-lite` could not express before (§3.9).

    Every other family, `refuse_invalid` included, is scoreable only by leaving a state
    footprint, so a fifth of RL training taught that out-of-policy requests warrant four to five
    tool calls — the opposite of what BFCL irrelevance scores.
    """
    spec = sample_task(0, ABSTAIN_FAMILY, "easy")
    good = run_episode(_silent_generator, spec, CONFIG)
    assert good.scores["r_outcome"] == 1.0
    assert good.scores["n_calls"] == 0.0
    # Full marks, so abstaining is competitive with solving any other family.
    assert good.scores["reward"] == pytest.approx(1.3)


def test_reading_before_declining_is_not_abstaining() -> None:
    """The regression the state check alone cannot catch.

    Reads leave the database untouched, so `db_hash` matches and the refusal text passes the
    output check: without folding restraint into `r_action`, four wasted lookups followed by a
    polite decline would score a perfect 1.0 on the family that exists to teach restraint.
    """
    spec = sample_task(0, ABSTAIN_FAMILY, "easy")
    result = run_episode(_one_read_then_decline, spec, CONFIG)
    assert result.scores["r_output"] == 1.0, "the refusal text itself was fine"
    assert result.scores["r_action"] == 0.0, "but calling anything must fail the task"
    assert result.scores["r_outcome"] == 0.0


def test_escalating_an_out_of_scope_request_is_not_abstaining() -> None:
    """Scored the way BFCL scores irrelevance: any call is wrong, `transfer_to_human` included.

    Deliberately stricter than the system prompt, whose escalation clause covers policy-forbidden
    *actions* rather than out-of-scope requests. The prompt is left untouched so every previously
    measured arm stays comparable, which means this distinction lives in the task, not the policy.
    """
    spec = sample_task(0, ABSTAIN_FAMILY, "easy")
    assert run_episode(_escalating_generator, spec, CONFIG).scores["r_outcome"] == 0.0


def test_saying_nothing_useful_is_not_abstaining_either() -> None:
    """Guards the mirror-image hack that created §3.9 in the first place.

    An abstention task's correct state is the untouched database, which a model that does nothing
    at all also produces. That is exactly how passivity once scored 1.0 on `refuse_invalid`. The
    fix there was to require a write; here it is the output check, so restraint stays expressible
    without making a state footprint mandatory.
    """
    spec = sample_task(0, ABSTAIN_FAMILY, "easy")
    result = run_episode(_cheerful_generator, spec, CONFIG)
    assert result.scores["n_calls"] == 0.0, "it did abstain from calling"
    assert result.scores["r_outcome"] == 0.0, "but never actually declined"


def test_abstention_topics_are_held_out_between_train_and_test() -> None:
    """Otherwise the in-domain score cannot tell restraint from having memorised 24 strings."""
    train = {sample_task(s, ABSTAIN_FAMILY, "easy").instruction for s in range(2000)}
    test = {sample_task(s, ABSTAIN_FAMILY, "easy").instruction for s in TEST_SEEDS}
    assert test and not (train & test)


def test_headline_eval_split_is_unchanged_by_the_abstention_family() -> None:
    """Comparability guard: `pass^1` must keep meaning what it meant for every reported arm."""
    assert EVAL_FAMILIES == FAMILIES
    assert ABSTAIN_FAMILY not in EVAL_FAMILIES
    assert len(build_specs(100, EVAL_FAMILIES, "easy")) == 600


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
