"""Build the SFT corpus from APIGen-MT-5k, optionally mixed with Hermes function-calling.

APIGen-MT ships 5k verified multi-turn trajectories in the tau-bench retail and airline
domains, in a ShareGPT-like schema:

    conversations: [{from: human|gpt|function_call|observation, value: str}]
    tools:         JSON string of {name, description, parameters}
    system:        the domain policy

The domain overlap with tau-retail-lite is the reason this dataset was chosen: SFT teaches
the multi-turn tool protocol in a neighbouring domain without ever touching the RL tasks.

That overlap turned out to have a cost. APIGen-MT is *strictly one tool call per assistant
turn* — 16,732 tool-calling messages, none with more than one — and so is `tau-retail-lite`,
so the model never saw two calls in a single turn in either stage. On BFCL it consequently
scored exactly 0/200 on `parallel` and `parallel_multiple`, having learned "emit one call and
stop" as a hard rule, while the base model manages 0.725 (WRITEUP.md §3.6).

`--hermes N` mixes in NousResearch/hermes-function-calling-v1, chosen over xLAM (gated) and
ToolACE (a bracketed DSL with spaces in function names, so it needs a bespoke parser) because
it needs neither: its schema is already OpenAI-style `tools` plus ShareGPT turns using the same
`<tool_call>` JSON that Qwen3's template emits. It supplies exactly what APIGen-MT lacks —
**56.8% of its tool-calling turns carry more than one call** (up to 10), across 35 domains
rather than one.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path
from typing import Any

DATA_URL = "https://huggingface.co/datasets/Salesforce/APIGen-MT-5k/resolve/main/apigen-mt_5k.json"

HERMES_DATASET = "NousResearch/hermes-function-calling-v1"
HERMES_CONFIG = "func_calling"

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)

# APIGen-MT exposes a `think` pseudo-tool. tau-retail-lite does not, and teaching the model
# to call tools that will not exist at evaluation time costs reward via the unknown-tool
# penalty, so these calls (and their observations) are dropped.
DROPPED_TOOLS = {"think"}


def download(cache: Path) -> Path:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        print(f"downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, cache)
    return cache


def convert_tools(raw: str) -> list[dict[str, Any]]:
    """ShareGPT tool list to the OpenAI-style schema chat templates expect."""
    tools = []
    for tool in json.loads(raw):
        if tool.get("name") in DROPPED_TOOLS:
            continue
        parameters = tool.get("parameters") or {"type": "object", "properties": {}}
        if "type" not in parameters:
            parameters = {"type": "object", "properties": parameters}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": parameters,
                },
            }
        )
    return tools


def convert_conversation(conversation: list[dict[str, str]]) -> list[dict[str, Any]] | None:
    """Map ShareGPT turns onto chat roles, dropping calls to tools we removed."""
    messages: list[dict[str, Any]] = []
    skip_next_observation = False

    for turn in conversation:
        role, value = turn["from"], turn["value"]

        if role == "human":
            skip_next_observation = False
            messages.append({"role": "user", "content": value})

        elif role == "gpt":
            skip_next_observation = False
            messages.append({"role": "assistant", "content": value})

        elif role == "function_call":
            try:
                call = json.loads(value)
            except json.JSONDecodeError:
                return None
            if call.get("name") in DROPPED_TOOLS:
                skip_next_observation = True
                continue
            skip_next_observation = False
            arguments = call.get("arguments", {})
            if not isinstance(arguments, dict):
                return None
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"type": "function", "function": {"name": call["name"], "arguments": arguments}}],
                }
            )

        elif role == "observation":
            if skip_next_observation:
                skip_next_observation = False
                continue
            if not messages or messages[-1].get("tool_calls") is None:
                return None  # an observation with no preceding call: malformed
            messages.append({"role": "tool", "content": value})

    return messages


def convert_hermes_conversation(conversation: list[dict[str, str]]) -> list[dict[str, Any]] | None:
    """Map a Hermes trajectory onto chat roles, preserving multi-call assistant turns.

    Preserving them is the entire point: an assistant turn holding three `<tool_call>` blocks
    becomes one message with three entries in `tool_calls`, not three messages. Flattening it
    would reproduce the exact defect this mix exists to fix.

    Hermes' own system turn is dropped. It is a generic instruction with the tool schemas
    inlined in `<tools>` tags, and the schemas are already passed to the chat template via the
    `tools` argument — keeping both would render every signature twice and, worse, in a layout
    the evaluation harness never uses.
    """
    messages: list[dict[str, Any]] = []

    for turn in conversation:
        role, value = turn["from"], turn["value"]

        if role == "system":
            continue
        if role == "human":
            messages.append({"role": "user", "content": value})

        elif role == "gpt":
            calls = []
            for payload in TOOL_CALL_RE.findall(value):
                try:
                    call = json.loads(payload)
                except json.JSONDecodeError:
                    return None
                arguments = call.get("arguments", {})
                if not isinstance(arguments, dict) or "name" not in call:
                    return None
                calls.append({"type": "function", "function": {"name": call["name"], "arguments": arguments}})
            if calls:
                messages.append({"role": "assistant", "content": "", "tool_calls": calls})
            else:
                messages.append({"role": "assistant", "content": value})

        elif role == "tool":
            responses = TOOL_RESPONSE_RE.findall(value)
            if not responses:
                return None
            if not messages or messages[-1].get("tool_calls") is None:
                return None  # a response with no preceding call: malformed
            # One `tool` turn answers all the calls of the preceding turn, so it expands into
            # one message per response — which is what the template expects to see.
            for response in responses:
                messages.append({"role": "tool", "content": response})

    return messages


def load_hermes(limit: int) -> list[dict[str, Any]]:
    """Hermes trajectories in the same `{messages, tools}` schema as the APIGen ones."""
    from datasets import load_dataset

    raw = load_dataset(HERMES_DATASET, HERMES_CONFIG, split="train")
    print(f"loaded {len(raw)} hermes trajectories")

    kept, skipped = [], 0
    for example in raw:
        tools = example["tools"]
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except json.JSONDecodeError:
                skipped += 1
                continue
        messages = convert_hermes_conversation(example["conversations"])
        if messages is None or not tools:
            skipped += 1
            continue
        if not any(m.get("tool_calls") for m in messages):
            skipped += 1
            continue
        kept.append({"messages": messages, "tools": tools})
        if len(kept) >= limit:
            break

    parallel = sum(
        1 for r in kept for m in r["messages"] if len(m.get("tool_calls") or []) > 1
    )
    print(f"hermes: kept {len(kept)} ({skipped} skipped), {parallel} multi-call assistant turns")
    return kept


def build(limit: int | None, out_path: Path, cache: Path, hermes: int = 0, seed: int = 0) -> None:
    raw = json.loads(download(cache).read_text())
    print(f"loaded {len(raw)} raw trajectories")

    kept, skipped = [], 0
    for example in raw:
        messages = convert_conversation(example["conversations"])
        tools = convert_tools(example["tools"])
        if messages is None or not tools:
            skipped += 1
            continue
        # Require a real multi-turn tool trajectory: at least one call and one user turn.
        if not any(m.get("tool_calls") for m in messages):
            skipped += 1
            continue
        if example.get("system"):
            messages = [{"role": "system", "content": example["system"]}] + messages
        kept.append({"messages": messages, "tools": tools})
        if limit and len(kept) >= limit:
            break

    print(f"apigen: kept {len(kept)} trajectories ({skipped} skipped)")

    if hermes:
        kept += load_hermes(hermes)
        # Interleave, so a truncated or interrupted run still sees both sources rather than
        # training on all of one and none of the other.
        random.Random(seed).shuffle(kept)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for record in kept:
            handle.write(json.dumps(record) + "\n")

    call_turns = [m for r in kept for m in r["messages"] if m.get("tool_calls")]
    multi = sum(1 for m in call_turns if len(m["tool_calls"]) > 1)
    print(f"total {len(kept)} trajectories")
    print(f"mean tool calls/trajectory: {len(call_turns) / max(1, len(kept)):.2f}")
    print(f"assistant turns with tool calls: {len(call_turns)}")
    print(f"  ...with more than one call: {multi} ({100 * multi / max(1, len(call_turns)):.1f}%)")
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/sft_apigen.jsonl"))
    parser.add_argument("--cache", type=Path, default=Path("data/apigen-mt_5k.json"))
    parser.add_argument(
        "--hermes",
        type=int,
        default=0,
        help="mix in this many Hermes trajectories (diverse domains, parallel calls)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build(args.limit, args.out, args.cache, args.hermes, args.seed)


if __name__ == "__main__":
    main()
