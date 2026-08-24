"""Pull evaluation summaries out of the Modal volume and render the comparison table.

Usage:
    modal volume get tooluse-workspace results ./results --force
    python scripts/collect_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

# `rl_only` is the ablation: an identically-configured LoRA trained with GRPO straight from
# the base model, so the SFT column and the RL-only column differ in one thing only.
TAGS = [
    ("base", "Base"),
    ("sft", "+ SFT"),
    ("rl_only", "GRPO only"),
    ("grpo", "+ SFT + GRPO"),
    ("rl_only_long", "GRPO only (long)"),
    ("grpo_long", "+ SFT + GRPO (long)"),
    # The mixed arms replace 500 of SFT's 1,000 APIGen trajectories with Hermes ones to restore
    # parallel calling (§3.7). `grpo_mixed` trains 30 GRPO steps from `sft_mixed`, matching `grpo`
    # step for step, so the two RL columns differ only in which prior they started from.
    ("sft_mixed", "+ SFT mixed"),
    ("grpo_mixed", "+ SFT mixed + GRPO"),
]

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
    ("lookup_compliance", "looked up before write", "{:.3f}"),
]


# Families whose oracle is a single write. Solving one *correctly* still requires looking the
# user up first, because the policy says so — but the reward never checks that, so the metric
# below has to.
WRITE_FAMILIES = {"cancel_order", "modify_address", "return_items", "exchange_items"}


def lookup_compliance(results_dir: Path, tag: str) -> float | None:
    """Fraction of single-write tasks where the model did anything before firing the write.

    This exists because success and compliance came apart. A checkpoint that skips
    `find_user_id_by_email` and calls the write directly with the id leaked in the prompt
    scores a *perfect* grounded reward while violating the first line of the policy. Success
    rising while this falls is the signature of that hack, and neither number shows it alone.
    """
    path = results_dir / f"{tag}_episodes.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    writes = [r for r in rows if r["family"] in WRITE_FAMILIES]
    if not writes:
        return None
    return sum(1 for r in writes if r["scores"]["n_calls"] > 1) / len(writes)


def load(results_dir: Path) -> dict[str, dict]:
    summaries = {}
    for tag, _ in TAGS:
        path = results_dir / f"{tag}_summary.json"
        if path.exists():
            summary = json.loads(path.read_text())
            compliance = lookup_compliance(results_dir, tag)
            if compliance is not None:
                summary["lookup_compliance"] = compliance
            summaries[tag] = summary
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
