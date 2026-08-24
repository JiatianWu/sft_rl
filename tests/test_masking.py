"""Checks that SFT sequences match what the model sees at inference.

A mismatch between the training rendering and the generation prompt is invisible in the
loss curve and silently degrades every downstream number, so it is asserted directly.
"""

from __future__ import annotations

import pytest

from tooluse.data.masking import (
    GENERATION_PREFIX,
    IGNORE_INDEX,
    build_example,
    generation_prompt,
    render_prefix,
)

MODEL = "Qwen/Qwen3-0.6B"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": "Get an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "The order id."}},
                "required": ["order_id"],
            },
        },
    }
]

MESSAGES = [
    {"role": "system", "content": "You are a retail agent."},
    {"role": "user", "content": "Where is order #W1?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "get_order_details", "arguments": {"order_id": "#W1"}}}],
    },
    {"role": "tool", "content": '{"status": "delivered"}'},
    {"role": "assistant", "content": "Your order #W1 has been delivered."},
]


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(MODEL)


def test_inference_prompt_differs_from_training_only_by_the_think_prefix(tokenizer) -> None:
    """Pin down the exact train/inference gap.

    The gap is deliberate and documented in `masking`: thinking is suppressed at
    inference so the base checkpoint is not handicapped. What must not happen is the gap
    silently *growing* if the chat template changes, so it is asserted exactly.
    """
    for index, message in enumerate(MESSAGES):
        if message["role"] != "assistant":
            continue
        training_prefix = render_prefix(tokenizer, MESSAGES[:index], TOOLS)
        inference_prompt = generation_prompt(tokenizer, MESSAGES[:index], TOOLS)
        assert inference_prompt == training_prefix + GENERATION_PREFIX, f"turn {index} diverges"


def test_history_never_contains_think_blocks(tokenizer) -> None:
    """Past assistant turns must render identically in training and during a rollout."""
    assert "<think>" not in render_prefix(tokenizer, MESSAGES, TOOLS)


def test_rendering_is_append_only(tokenizer) -> None:
    previous = ""
    for index in range(len(MESSAGES)):
        current = render_prefix(tokenizer, MESSAGES[: index + 1], TOOLS)
        assert current.startswith(previous)
        previous = current


def test_only_assistant_tokens_are_labelled(tokenizer) -> None:
    example = build_example(tokenizer, MESSAGES, TOOLS)
    assert example is not None
    input_ids, labels = example["input_ids"], example["labels"]
    assert len(input_ids) == len(labels)

    supervised = tokenizer.decode([i for i, l in zip(input_ids, labels) if l != IGNORE_INDEX])
    # The assistant's tool call and final answer are supervised.
    assert "get_order_details" in supervised
    assert "has been delivered" in supervised
    # The tool's output and the user's question are not.
    assert "Where is order" not in supervised
    assert '"status": "delivered"' not in supervised


def test_tool_output_is_masked_out(tokenizer) -> None:
    """Training on tool results would teach the model to invent database contents."""
    example = build_example(tokenizer, MESSAGES, TOOLS)
    masked = tokenizer.decode(
        [i for i, l in zip(example["input_ids"], example["labels"]) if l == IGNORE_INDEX]
    )
    assert "tool_response" in masked


def test_truncation_drops_uninformative_examples(tokenizer) -> None:
    assert build_example(tokenizer, MESSAGES, TOOLS, max_length=8) is None


PARALLEL_MESSAGES = [
    {"role": "user", "content": "Where are orders #W1 and #W2?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": "get_order_details", "arguments": {"order_id": "#W1"}}},
            {"type": "function", "function": {"name": "get_order_details", "arguments": {"order_id": "#W2"}}},
        ],
    },
    {"role": "tool", "content": '{"order_id": "#W1", "status": "delivered"}'},
    {"role": "tool", "content": '{"order_id": "#W2", "status": "pending"}'},
    {"role": "assistant", "content": "#W1 is delivered and #W2 is pending."},
]


def test_a_turn_calling_two_tools_at_once_survives_masking(tokenizer) -> None:
    """Parallel calls must produce a usable training example, not be silently discarded.

    Qwen3's template merges consecutive tool messages into one user turn, so rendering after
    each response individually is not append-only and the prefix check rejects the trajectory.
    Every message here is masked identically either way, so `_segment_ends` emits the run as a
    single unit.

    This is not hypothetical tidiness. Mixing in Hermes to teach parallel calling dropped all
    284 of its multi-call trajectories — precisely the examples being added — and the corpus
    would have trained on zero of them while reporting a full 1,000. Nothing downstream would
    have shown it: the loss curve is normal and the arm simply fails to learn the thing it was
    built to learn.
    """
    example = build_example(tokenizer, PARALLEL_MESSAGES, TOOLS)
    assert example is not None, "a two-call assistant turn was dropped by the masker"

    supervised = tokenizer.decode(
        [i for i, l in zip(example["input_ids"], example["labels"]) if l != IGNORE_INDEX]
    )
    # Both calls are supervised, so the model is taught to emit two in one turn.
    assert supervised.count("get_order_details") == 2
    assert "#W1" in supervised and "#W2" in supervised
    # Both tool responses stay masked.
    assert '"status": "delivered"' not in supervised
    assert '"status": "pending"' not in supervised


def test_rendering_is_append_only_across_a_run_of_tool_messages(tokenizer) -> None:
    """The segmenting, not the template, is what makes the parallel case prefix-stable."""
    from tooluse.data.masking import _segment_ends

    assert _segment_ends(PARALLEL_MESSAGES) == [0, 1, 3, 4]  # the two tool turns collapse to one
    previous = ""
    for index in _segment_ends(PARALLEL_MESSAGES):
        current = render_prefix(tokenizer, PARALLEL_MESSAGES[: index + 1], TOOLS)
        assert current.startswith(previous), f"segment ending at {index} broke prefix stability"
        previous = current
