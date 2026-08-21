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
