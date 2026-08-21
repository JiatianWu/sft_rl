"""Convert APIGen-MT-5k into chat-format multi-turn tool-use trajectories.

APIGen-MT ships 5k verified multi-turn trajectories in the tau-bench retail and airline
domains, in a ShareGPT-like schema:

    conversations: [{from: human|gpt|function_call|observation, value: str}]
    tools:         JSON string of {name, description, parameters}
    system:        the domain policy

The domain overlap with tau-retail-lite is the reason this dataset was chosen: SFT teaches
the multi-turn tool protocol in a neighbouring domain without ever touching the RL tasks.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

DATA_URL = "https://huggingface.co/datasets/Salesforce/APIGen-MT-5k/resolve/main/apigen-mt_5k.json"

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


def build(limit: int | None, out_path: Path, cache: Path) -> None:
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for record in kept:
            handle.write(json.dumps(record) + "\n")

    n_calls = sum(sum(1 for m in r["messages"] if m.get("tool_calls")) for r in kept)
    print(f"kept {len(kept)} trajectories ({skipped} skipped)")
    print(f"mean tool calls/trajectory: {n_calls / max(1, len(kept)):.2f}")
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/sft_apigen.jsonl"))
    parser.add_argument("--cache", type=Path, default=Path("data/apigen-mt_5k.json"))
    args = parser.parse_args()
    build(args.limit, args.out, args.cache)


if __name__ == "__main__":
    main()
