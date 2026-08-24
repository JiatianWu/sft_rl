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

# Inference-time template settings, imported by RL and eval so the stages cannot drift.
#
# `enable_thinking=False` makes the generation prompt end with an empty
# `<think>\n\n</think>\n\n` block, which suppresses chain-of-thought. This is not a
# stylistic choice: with thinking enabled, Qwen3-0.6B spends its entire completion budget
# reasoning and never emits a tool call, which would handicap the *base* checkpoint
# specifically and inflate the apparent gain from SFT.
#
# Note the one residual train/inference difference this leaves. During a rollout the
# think block is ephemeral: it prefixes the turn being generated, but once the tool result
# is appended and the conversation is re-rendered, past assistant turns appear without it.
# A single contiguous SFT sequence cannot reproduce that (it would have think blocks on
# every assistant turn or none). Training without them keeps the mismatch to a constant
# 5-token prefix on the current turn instead of spreading it across the whole history.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

# What `add_generation_prompt=True` appends beyond a training prefix, under the settings
# above. Asserted in the tests so a template change cannot silently widen the gap.
GENERATION_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


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


def _segment_ends(messages: list[dict[str, Any]]) -> list[int]:
    """Indices to render at, treating a run of consecutive tool messages as one unit.

    Qwen3's template merges consecutive tool messages into a *single* user turn, so rendering
    after each one is not append-only: with one response the block closes with
    `</tool_response><|im_end|>`, and adding the second reopens it. The incremental diff then
    fails its prefix check and `build_example` discards the trajectory.

    That only bites when an assistant turn makes several calls at once, which no single-call
    corpus contains — so this went unnoticed until Hermes was mixed in, where it silently
    dropped *all* 284 parallel trajectories, i.e. exactly the examples the mix exists to add.

    Emitting the run as one segment is safe rather than merely convenient: every message in it
    is a tool response, so the whole span is masked either way and per-message granularity buys
    nothing.
    """
    ends = []
    for index, message in enumerate(messages):
        following = messages[index + 1] if index + 1 < len(messages) else None
        if message["role"] == "tool" and following is not None and following["role"] == "tool":
            continue
        ends.append(index)
    return ends


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

    for index in _segment_ends(messages):
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


__all__ = [
    "CHAT_TEMPLATE_KWARGS",
    "GENERATION_PREFIX",
    "IGNORE_INDEX",
    "build_example",
    "generation_prompt",
    "render_prefix",
]
