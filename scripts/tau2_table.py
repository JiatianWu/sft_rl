"""Aggregate τ-bench retail results across arms.

The reason this is a script rather than a printed average: τ-bench leaves `reward_basis`,
`db_check` and `communicate_checks` **null** when a simulation ends on `max_steps` or an error,
while `reward` still defaults to `0.0`. Averaging over every simulation therefore manufactures a
clean `pass^1 = 0.000` out of runs that were never scored at all — a number that reads as a
capability verdict but is really an accounting mistake. Scored and unscored are separated here,
and `pass^1` is reported as `n/a` when nothing reached an evaluator.

Two columns exist purely to keep the headline honest:

- **normal-stop rate** is the health check on the customer. TAU2_PREREGISTRATION.md records that a
  weak user simulator penalises *asking* far more than *acting*, which is the exact axis P18
  measures, so a low normal-stop rate means the comparison is untrustworthy in both directions.
- **DB and COMMUNICATE apart**, because SFT could gain on COMMUNICATE alone by being chattier,
  and that would look like the §3.1 reversal without being it.
"""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path

ARMS = [
    ("base", "Base"),
    ("sft", "+ SFT"),
    ("grpo", "SFT+RL (30)"),
    ("rl_only_long", "RL only (200)"),
    ("sft_1500", "SFT 1500"),
    ("grpo_1500", "SFT 1500 + RL (30)"),
]

RESULTS = Path("results/tau2")


def wilson(correct: int, total: int) -> tuple[float, float]:
    """95% interval. Small denominators are the norm here, so normal approximation will not do."""
    if total == 0:
        return (0.0, 0.0)
    z = 1.96
    p = correct / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def load(tag: str) -> dict | None:
    path = RESULTS / f"{tag}.json"
    if not path.exists():
        return None
    simulations = json.loads(path.read_text()).get("simulations", [])
    if not simulations:
        return None

    terminations = collections.Counter(s.get("termination_reason") or "unknown" for s in simulations)
    scored = [s for s in simulations if (s.get("reward_info") or {}).get("reward_basis") is not None]
    rewards = [(s["reward_info"]).get("reward", 0.0) for s in scored]

    components: dict[str, list[float]] = collections.defaultdict(list)
    for simulation in scored:
        for key, value in ((simulation["reward_info"]).get("reward_breakdown") or {}).items():
            if isinstance(value, (int, float)):
                components[str(key)].append(float(value))

    return {
        "n": len(simulations),
        "scored": len(scored),
        "solved": sum(1 for r in rewards if r >= 1.0),
        "pass1": (sum(rewards) / len(rewards)) if rewards else None,
        "normal_stop": terminations.get("user_stop", 0) + terminations.get("agent_stop", 0),
        "terminations": dict(terminations),
        "components": {k: sum(v) / len(v) for k, v in components.items() if v},
    }


def main() -> None:
    rows = [(tag, label, load(tag)) for tag, label in ARMS]
    rows = [(tag, label, data) for tag, label, data in rows if data]
    if not rows:
        print(f"no τ-bench results under {RESULTS}/")
        return

    print("τ-bench retail\n")
    header = f"{'arm':<22} {'sims':>5} {'scored':>7} {'pass^1':>16} {'solved':>7} {'normal stop':>12}"
    print(header)
    print("-" * len(header))
    for _, label, data in rows:
        if data["pass1"] is None:
            score = "n/a"
        else:
            low, high = wilson(data["solved"], data["scored"])
            score = f"{data['pass1']:.3f} [{low:.2f},{high:.2f}]"
        stop = f"{data['normal_stop']}/{data['n']}"
        print(f"{label:<22} {data['n']:>5} {data['scored']:>7} {score:>16} {data['solved']:>7} {stop:>12}")

    print("\nreward components (scored simulations only)")
    for _, label, data in rows:
        if data["components"]:
            parts = "  ".join(f"{k}={v:.3f}" for k, v in sorted(data["components"].items()))
            print(f"  {label:<22} {parts}")
        else:
            print(f"  {label:<22} (nothing scored)")

    print("\ntermination reasons")
    for _, label, data in rows:
        print(f"  {label:<22} {data['terminations']}")

    total_scored = sum(d["scored"] for _, _, d in rows)
    if total_scored == 0:
        print(
            "\nNothing was scored in any arm, so this run discriminates nothing. "
            "TAU2_PREREGISTRATION.md commits to reporting that as "
            "'below the measurement floor of τ-bench retail for a 0.6B model' "
            "rather than as pass^1 = 0.000."
        )


if __name__ == "__main__":
    main()
