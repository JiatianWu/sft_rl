"""Aggregate BFCL scores and test whether the arms differ at all.

The per-category accuracies are close enough that eyeballing them invites reading noise as
signal, which is exactly the failure this benchmark was run to avoid. So every comparison here
carries an interval, and the summary states which differences survive it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ARMS = [
    ("base", "Base"),
    ("sft", "+ SFT"),
    ("grpo", "SFT+RL (30)"),
    ("rl_only_long", "RL only (200)"),
    ("sft_mixed", "SFT mixed"),
    ("grpo_mixed", "SFT mixed + RL (30)"),
    ("grpo_abstain", "+ abstention family"),
    ("sft_1500", "SFT 1500"),
    ("grpo_1500", "SFT 1500 + RL (30)"),
]

# Grouped the way the pre-registration reads them: syntax, restraint, and the multi-turn
# category that P1 rests on. BFCL's own V4 `Overall` is deliberately not reported — 40% of its
# weight is agentic web search and long-session memory, which this pipeline never trained.
GROUPS = {
    "AST (syntax)": ["simple_python", "multiple", "parallel", "parallel_multiple"],
    "Restraint": ["irrelevance", "live_irrelevance", "live_relevance"],
    "Multi-turn": ["multi_turn_base"],
}


SUMMARY = Path("results/bfcl_summary.json")


def load(root: Path, tag: str) -> dict[str, tuple[int, int]]:
    """Per-category (correct, total) for one arm.

    Prefers the raw BFCL score files, and falls back to the committed summary. The raw files are
    50 MB because every failure record embeds its full prompt, which is too much to keep in git —
    but the counts are the whole of what the tables need, so they are committed separately and
    the numbers stay reproducible from a fresh clone.
    """
    counts = {}
    directory = root / tag
    if directory.exists():
        for path in directory.rglob("*_score.json"):
            summary = json.loads(path.read_text().splitlines()[0])
            name = path.stem.replace("BFCL_v4_", "").replace("_score", "")
            counts[name] = (summary["correct_count"], summary["total_count"])
    if not counts and SUMMARY.exists():
        stored = json.loads(SUMMARY.read_text()).get(root.name, {}).get(tag, {})
        counts = {name: tuple(pair) for name, pair in stored.items()}
    return counts


def write_summary() -> None:
    """Persist the counts for both runs so the tables survive without the 50 MB of raw output."""
    payload = {}
    for root in (Path("results/bfcl"), Path("results/bfcl_noise_floor")):
        if not root.exists():
            continue
        payload[root.name] = {tag: load(root, tag) for tag, _ in ARMS}
    if payload:
        SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def wilson(correct: int, total: int) -> tuple[float, float]:
    """Wilson 95% interval — behaves at the ~5% accuracies where multi-turn lives.

    The normal approximation is unreliable that close to zero, and multi-turn is the category
    the load-bearing prediction depends on.
    """
    if total == 0:
        return (0.0, 0.0)
    z, p, n = 1.96, correct / total, total
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def two_proportion_p(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Two-sided p-value for two proportions, normal approximation."""
    (c1, n1), (c2, n2) = a, b
    if n1 == 0 or n2 == 0:
        return 1.0
    pooled = (c1 + c2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = abs(c1 / n1 - c2 / n2) / se
    return math.erfc(z / math.sqrt(2))


def main() -> None:
    write_summary()
    root = Path("results/bfcl")
    data = {tag: load(root, tag) for tag, _ in ARMS}

    print("\nBFCL per category (accuracy, n)\n")
    header = f"{'category':<22}" + "".join(f"{label:>18}" for _, label in ARMS)
    print(header)
    print("-" * len(header))
    for group, categories in GROUPS.items():
        print(f"\n{group}")
        for category in categories:
            row = f"  {category:<20}"
            for tag, _ in ARMS:
                correct, total = data[tag].get(category, (0, 0))
                row += f"{correct / total:>13.3f} ({total})" if total else f"{'—':>18}"
            print(row)

    print("\n\nPooled by group, with 95% Wilson intervals\n")
    print(f"{'group':<16}" + "".join(f"{label:>24}" for _, label in ARMS))
    print("-" * (16 + 24 * len(ARMS)))
    pooled: dict[str, dict[str, tuple[int, int]]] = {}
    for group, categories in GROUPS.items():
        row = f"{group:<16}"
        pooled[group] = {}
        for tag, _ in ARMS:
            correct = sum(data[tag].get(c, (0, 0))[0] for c in categories)
            total = sum(data[tag].get(c, (0, 0))[1] for c in categories)
            pooled[group][tag] = (correct, total)
            low, high = wilson(correct, total)
            row += f"{correct / total:>12.3f} [{low:.3f},{high:.3f}]" if total else f"{'—':>24}"
        print(row)

    print("\n\nDoes any arm differ from base?\n")
    for group in GROUPS:
        base = pooled[group]["base"]
        for tag, label in ARMS[1:]:
            arm = pooled[group][tag]
            delta = arm[0] / arm[1] - base[0] / base[1]
            p = two_proportion_p(arm, base)
            verdict = "significant" if p < 0.05 else "within noise"
            print(f"  {group:<14} {label:<16} {delta:+.3f}  p={p:.2f}  {verdict}")

    print("\n\nIs 'restraint' judgement, or just a bias against calling?\n")
    # The restraint group is 1,124 cases where the right move is to stay silent against 16 where
    # it is to call. A model that simply stopped calling functions would top that group while
    # being useless, so the two halves have to be read separately.
    for tag, label in ARMS:
        abstain_c = sum(data[tag].get(c, (0, 0))[0] for c in ["irrelevance", "live_irrelevance"])
        abstain_n = sum(data[tag].get(c, (0, 0))[1] for c in ["irrelevance", "live_irrelevance"])
        call_c, call_n = data[tag].get("live_relevance", (0, 0))
        print(
            f"  {label:<16} should NOT call: {abstain_c / abstain_n:.3f} ({abstain_n})"
            f"   should call: {call_c / call_n:.3f} ({call_n})"
        )

    print("\n\nNoise floor: four byte-identical models, scored independently\n")
    # A merge bug briefly produced four copies of the base weights, and the sweep was run on them
    # before it was caught. That accident measures BFCL's run-to-run spread directly: at
    # temperature 0.001 the only variation is vLLM batching non-determinism. Differences smaller
    # than this band mean nothing, which is what sinks P1.
    noise = {tag: load(Path("results/bfcl_noise_floor"), tag) for tag, _ in ARMS}
    for group, categories in GROUPS.items():
        rates = []
        for tag, _ in ARMS:
            correct = sum(noise[tag].get(c, (0, 0))[0] for c in categories)
            total = sum(noise[tag].get(c, (0, 0))[1] for c in categories)
            if total:
                rates.append(correct / total)
        if rates:
            print(f"  {group:<14} {min(rates):.3f} – {max(rates):.3f}  (spread {max(rates) - min(rates):.3f})")

    print("\n\nP1: the pre-registered prediction\n")
    a, b = pooled["Multi-turn"]["grpo"], pooled["Multi-turn"]["rl_only_long"]
    p = two_proportion_p(a, b)
    print(f"  SFT+RL (30)   {a[0]}/{a[1]} = {a[0] / a[1]:.3f}")
    print(f"  RL only (200) {b[0]}/{b[1]} = {b[0] / b[1]:.3f}")
    print(f"  difference {a[0] / a[1] - b[0] / b[1]:+.3f}, p={p:.2f}")
    print(
        "  Direction matches the prediction, magnitude is two test cases.\n"
        "  Verdict: UNRESOLVED — not support, and not refutation either."
    )


if __name__ == "__main__":
    main()
