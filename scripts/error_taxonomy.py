"""Classify failed episodes into a taxonomy, so an aggregate score points at something.

`r_outcome = 0.41` says a checkpoint is bad but not what to fix. The buckets below separate
failures the pipeline can act on (protocol errors, which SFT addresses) from failures it
cannot (wrong decisions, which need a better policy or a bigger model).

Each episode lands in exactly one bucket, assigned in the order the failure happens: an
episode that never emits a parseable call cannot also be judged on its arguments.

Usage:
    python scripts/error_taxonomy.py [results_dir]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

TAGS = [
    ("base", "Base"),
    ("sft", "+ SFT"),
    ("rl_only", "GRPO only"),
    ("grpo", "+ SFT + GRPO"),
    ("rl_only_long", "GRPO only (long)"),
    ("grpo_long", "+ SFT + GRPO (long)"),
]

# Ordered: the first matching rule wins, so each episode is counted once.
BUCKETS = [
    (
        "no tool call",
        lambda s, r: s["called_any_tool"] == 0,
        "answered from the prompt alone, never touched a tool",
    ),
    (
        "malformed / unknown tool",
        lambda s, r: s["n_malformed"] > 0 or s["n_unknown_tool"] > 0,
        "emitted something call-shaped that could not be executed",
    ),
    (
        "illegal write",
        lambda s, r: s["n_illegal_writes"] > 0,
        "called a write the policy or order state forbids",
    ),
    (
        "stopped early",
        lambda s, r: r["stopped_reason"] == "final_answer" and s["r_progress"] < 1.0,
        "answered the user with oracle actions still outstanding",
    ),
    (
        "ran out of turns",
        lambda s, r: r["stopped_reason"] == "max_turns",
        "kept calling tools until the turn limit",
    ),
    (
        "acted, reported badly",
        lambda s, r: s["r_action"] == 1.0 and s["r_output"] == 0.0,
        "reached the right database state but did not tell the user the required facts",
    ),
    (
        "reported, acted badly",
        lambda s, r: s["r_output"] == 1.0 and s["r_action"] == 0.0,
        "said the right thing without making it true",
    ),
    ("other", lambda s, r: True, "everything else"),
]


def classify(record: dict) -> str:
    scores = record["scores"]
    for name, rule, _ in BUCKETS:
        if rule(scores, record):
            return name
    return "other"


def taxonomy(records: list[dict]) -> tuple[Counter, int]:
    failures = [r for r in records if r["scores"]["r_outcome"] < 1.0]
    return Counter(classify(r) for r in failures), len(records)


def main() -> None:
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    tables: dict[str, tuple[Counter, int]] = {}
    for tag, _ in TAGS:
        path = results_dir / f"{tag}_episodes.jsonl"
        if path.exists():
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            tables[tag] = taxonomy(records)

    if not tables:
        raise SystemExit(f"no *_episodes.jsonl found in {results_dir}/")

    present = [(tag, label) for tag, label in TAGS if tag in tables]
    print("Share of **all** episodes, so columns are comparable across checkpoints.\n")
    print("| Failure mode | " + " | ".join(label for _, label in present) + " |")
    print("|---|" + "---|" * len(present))
    for name, _, _ in BUCKETS:
        cells = []
        for tag, _ in present:
            counts, total = tables[tag]
            cells.append(f"{counts.get(name, 0) / total:.1%}" if counts.get(name) else "-")
        if any(cell != "-" for cell in cells):
            print(f"| {name} | " + " | ".join(cells) + " |")
    cells = [f"**{1 - sum(tables[tag][0].values()) / tables[tag][1]:.1%}**" for tag, _ in present]
    print("| **solved** | " + " | ".join(cells) + " |")

    print("\nBucket meanings:\n")
    for name, _, description in BUCKETS:
        if any(name in tables[tag][0] for tag, _ in present):
            print(f"- **{name}** — {description}")


if __name__ == "__main__":
    main()
