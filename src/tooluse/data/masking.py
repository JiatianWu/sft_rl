"""Render trajectories and build assistant-only loss masks.

Two details of Qwen3's chat template, both verified by reading the template rather than
assuming, dictate the approach here:

1. There is no `{% generation %}` marker, so TRL's `assistant_only_loss` cannot be used.
2. An assistant message renders differently when it is the *last* message in the list:
   the template injects an empty `<think>\\n\\n</think>` block. That makes
   `render(messages[:i])` not a prefix of `render(messages[:j])`, so naive incremental
   diffing silently misaligns the mask. Appending a sentinel user turn before rendering
   removes the special case and restores prefix stability.

Masking matters beyond tidiness: training on tool-response tokens teaches the model to
*generate* database contents, which is exactly the hallucination we are trying to avoid.
"""

from __future__ import annotations

from typing import Any

SENTINEL = "__END_OF_TRAJECTORY_SENTINEL__"
IGNORE_INDEX = -100

# `enable_thinking=True` is the setting that produces a *bare* `<|im_start|>assistant\n`
# generation prompt. The opposite setting injects an empty think block that the training
# data would not contain. This constant is imported by training and eval so that the
# three stages cannot drift apart.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": True}


def render_prefix(tokenizer, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    """Render `messages` with every assistant turn in its non-final form."""
    text = tokenizer.apply_chat_template(
        list(messages) + [{"role": "user", "content": SENTINEL}],
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
        **CHAT_TEMPLATE_KWARGS,
    )
    marker = text.rindex("<|im_start|>user")
    # Guard against a trajectory whose own content contains the sentinel.
    if SENTINEL not in text[marker:]:
        raise ValueError("sentinel turn not found where expected")
    return text[:marker]


def build_example(
    tokenizer,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_length: int = 8192,
) -> dict[str, list[int]] | None:
    """Tokenise one trajectory, labelling only assistant tokens.

    Returns None when the trajectory cannot be represented faithfully (tokenisation is
    not prefix-stable, or nothing is left to learn from after truncation).
    """
    input_ids: list[int] = []
    labels: list[int] = []
    previous = ""

    for index in range(len(messages)):
        current = render_prefix(tokenizer, messages[: index + 1], tools)
        if not current.startswith(previous):
            return None  # rendering was not append-only; refuse to guess at the mask
        delta = current[len(previous) :]
        previous = current

        delta_ids = tokenizer(delta, add_special_tokens=False)["input_ids"]
        input_ids.extend(delta_ids)
        if messages[index]["role"] == "assistant":
            labels.extend(delta_ids)
        else:
            labels.extend([IGNORE_INDEX] * len(delta_ids))

    # Segment-wise tokenisation must agree with tokenising the whole string, otherwise
    # the mask is offset from the tokens it is supposed to align with.
    if tokenizer(previous, add_special_tokens=False)["input_ids"] != input_ids:
        return None

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    if not any(label != IGNORE_INDEX for label in labels):
        return None

    return {"input_ids": input_ids, "labels": labels}


def generation_prompt(tokenizer, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    """The exact prompt used at inference. Must equal a training prefix."""
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        **CHAT_TEMPLATE_KWARGS,
    )


__all__ = ["CHAT_TEMPLATE_KWARGS", "IGNORE_INDEX", "build_example", "generation_prompt", "render_prefix"]
