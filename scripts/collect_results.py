"""Pull evaluation summaries out of the Modal volume and render the comparison table.

Usage:
    modal volume get tooluse-workspace results ./results --force
    python scripts/collect_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

TAGS = [("base", "Base"), ("sft", "+ SFT"), ("grpo", "+ SFT + GRPO")]

HEADLINE = [
    ("pass^1", "pass^1", "{:.3f}"),
    ("pass^4", "pass^4", "{:.3f}"),
    ("r_action", "state ok", "{:.3f}"),
    ("r_output", "reported ok", "{:.3f}"),
    ("r_outcome", "success", "{:.3f}"),
    ("r_progress", "progress", "{:.3f}"),
    ("n_calls", "tool calls/ep", "{:.2f}"),
    ("called_any_tool", "used a tool", "{:.2f}"),
    ("n_illegal_writes", "illegal writes", "{:.3f}"),
    ("n_malformed", "malformed", "{:.3f}"),
]


def load(results_dir: Path) -> dict[str, dict]:
    summaries = {}
    for tag, _ in TAGS:
        path = results_dir / f"{tag}_summary.json"
        if path.exists():
            summaries[tag] = json.loads(path.read_text())
    return summaries


def table(summaries: dict[str, dict], rows: list[tuple[str, str, str]]) -> str:
    present = [(tag, label) for tag, label in TAGS if tag in summaries]
    header = "| Metric | " + " | ".join(label for _, label in present) + " |"
    divider = "|---|" + "---|" * len(present)
    lines = [header, divider]
    for key, label, fmt in rows:
        cells = []
        for tag, _ in present:
            value = summaries[tag].get(key)
            cells.append(fmt.format(value) if isinstance(value, (int, float)) else "-")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def family_table(summaries: dict[str, dict]) -> str:
    present = [(tag, label) for tag, label in TAGS if tag in summaries]
    families = sorted({f for s in summaries.values() for f in s.get("per_family_success", {})})
    header = "| Task family | " + " | ".join(label for _, label in present) + " |"
    lines = [header, "|---|" + "---|" * len(present)]
    for family in families:
        cells = [f"{summaries[tag]['per_family_success'].get(family, float('nan')):.2f}" for tag, _ in present]
        marker = " *(held out from RL)*" if family == "exchange_items" else ""
        lines.append(f"| `{family}`{marker} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    results_dir = Path("results")
    summaries = load(results_dir)
    if not summaries:
        raise SystemExit("no summaries found in results/ - run `modal volume get` first")

    print("## Headline\n")
    print(table(summaries, HEADLINE))
    print("\n## Per-family success (r_outcome)\n")
    print(family_table(summaries))
    print("\n## Run metadata\n")
    for tag, label in TAGS:
        if tag in summaries:
            s = summaries[tag]
            print(f"- **{label}**: {s['n_tasks']} tasks x {s['trials']} trials, T={s['temperature']}, {s['elapsed_s']}s")


if __name__ == "__main__":
    main()
