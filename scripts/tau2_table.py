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

    # P22/P23 denominators exclude infrastructure errors, declared in the pre-registration before
    # the split was known, because the pilot's infra rate was uneven across arms (1, 3, 0).
    usable = [s for s in simulations if s.get("termination_reason") != "infrastructure_error"]

    # P23 measures the claimed mechanism directly. "SFT loops less" is the symptom; the claim is
    # that it *asks the customer* for what it is missing instead of inventing it, so asking is
    # counted rather than inferred from how the conversation died.
    asked_turns = agent_turns = conversations_with_a_question = 0
    for simulation in usable:
        asked_here = False
        for message in simulation.get("messages") or []:
            if message.get("role") != "assistant":
                continue
            agent_turns += 1
            if not message.get("tool_calls") and "?" in (message.get("content") or ""):
                asked_turns += 1
                asked_here = True
        conversations_with_a_question += int(asked_here)

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
        "usable": len(usable),
        "looped": sum(1 for s in usable if s.get("termination_reason") == "too_many_errors"),
        "asked_any": conversations_with_a_question,
        "ask_rate": (asked_turns / agent_turns) if agent_turns else 0.0,
    }


def main() -> None:
    rows = [(tag, label, load(tag)) for tag, label in ARMS]
    rows = [(tag, label, data) for tag, label, data in rows if data]
    if not rows:
        print(f"no τ-bench results under {RESULTS}/")
        return

    # The headline is solved/usable, not solved/scored. τ-bench only scores a conversation that
    # terminated normally, so averaging over scored simulations silently conditions on surviving —
    # and survival is exactly what differs between arms here. On tasks 10-49 that inflates
    # grpo_1500 from 1/40 to 1/6, making the worst arm look tied for best. A conversation that
    # looped until the error limit did not solve its task, so it belongs in the denominator.
    print("τ-bench retail\n")
    header = (
        f"{'arm':<22} {'sims':>5} {'usable':>7} {'solved/usable':>21} "
        f"{'scored-only (biased)':>21} {'normal stop':>12}"
    )
    print(header)
    print("-" * len(header))
    for _, label, data in rows:
        usable = data["usable"]
        if usable:
            low, high = wilson(data["solved"], usable)
            honest = f"{data['solved']}/{usable} = {data['solved'] / usable:.3f} [{low:.2f},{high:.2f}]"
        else:
            honest = "n/a"
        biased = (
            f"{data['solved']}/{data['scored']} = {data['pass1']:.3f}"
            if data["scored"]
            else "n/a"
        )
        stop = f"{data['normal_stop']}/{data['n']}"
        print(f"{label:<22} {data['n']:>5} {usable:>7} {honest:>21} {biased:>21} {stop:>12}")

    print("\nP22 loop-to-death rate and P23 asking rate (infrastructure errors excluded)")
    header = f"{'arm':<22} {'usable':>7} {'looped':>16} {'asked at all':>16} {'ask rate':>10}"
    print(header)
    print("-" * len(header))
    for _, label, data in rows:
        n = data["usable"]
        loop = f"{data['looped']}/{n} ({data['looped'] / n:.2f})" if n else "n/a"
        asked = f"{data['asked_any']}/{n} ({data['asked_any'] / n:.2f})" if n else "n/a"
        print(f"{label:<22} {n:>7} {loop:>16} {asked:>16} {data['ask_rate']:>10.3f}")

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
