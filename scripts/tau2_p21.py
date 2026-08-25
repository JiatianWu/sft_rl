"""Adjudicate P21: does `grpo_1500` lose reward *specifically* on refusal tasks?

P21 predicted that the project's worst abstainer would write to the DB on tasks whose correct
answer is to decline, changing the hash and scoring 0 on precisely those tasks — "visible as a
per-task pattern, not just an aggregate". Its refutation condition was failures spread evenly
across task types.

Classifying by gold action list is what makes this checkable at all: a task is a refusal task iff
its `evaluation_criteria.actions` terminate in `transfer_to_human_agents`, which is τ-bench's
equivalent of the in-domain `refuse_invalid` family (both require an explicit escalating call
rather than a text-only decline — see WRITEUP §3.9).

The script prints the denominators first, because on this sample they are the finding.
"""

import json
from collections import Counter
from pathlib import Path

ARMS = ["base", "sft", "grpo", "grpo_1500"]
RESULTS = Path(__file__).resolve().parents[1] / "results" / "tau2"

# Every retail tool that mutates the DB, i.e. every call that can move the hash `db_check` compares.
WRITES = {
    "cancel_pending_order",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
}
ESCALATE = "transfer_to_human_agents"


def classify(task: dict) -> str:
    """refusal | write | readonly, from the gold action list."""
    names = [a["name"] for a in (task.get("evaluation_criteria") or {}).get("actions") or []]
    if ESCALATE in names:
        return "refusal"
    return "write" if set(names) & WRITES else "readonly"


def main() -> None:
    kinds: dict[str, str] = {}
    per_arm: dict[str, dict[str, list[dict]]] = {}

    for arm in ARMS:
        data = json.loads((RESULTS / f"{arm}.json").read_text())
        for task in data["tasks"]:
            kinds[str(task["id"])] = classify(task)

        buckets: dict[str, list[dict]] = {}
        for sim in data["simulations"]:
            task_id = str(sim["task_id"])
            calls = [
                call["name"]
                for message in sim["messages"]
                if message.get("role") == "assistant"
                for call in (message.get("tool_calls") or [])
            ]
            info = sim.get("reward_info") or {}
            buckets.setdefault(kinds[task_id], []).append(
                {
                    "task": task_id,
                    "scored": info.get("reward_basis") is not None,
                    "reward": info.get("reward", 0.0),
                    "wrote": any(name in WRITES for name in calls),
                    "escalated": ESCALATE in calls,
                    "looped": sim.get("termination_reason") == "too_many_errors",
                }
            )
        per_arm[arm] = buckets

    counts = Counter(kinds.values())
    print(f"tasks by kind: {dict(counts)}   (ids 10-49)")
    print(f"refusal tasks: {sorted(t for t, k in kinds.items() if k == 'refusal')}\n")

    header = f"{'arm':<11}" + "".join(f"{k:>26}" for k in ("refusal", "write", "readonly"))
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        cells = []
        for kind in ("refusal", "write", "readonly"):
            rows = per_arm[arm].get(kind, [])
            if not rows:
                cells.append(f"{'-':>26}")
                continue
            solved = sum(1 for r in rows if r["reward"] >= 1.0)
            wrote = sum(1 for r in rows if r["wrote"])
            cells.append(f"{f'{solved}/{len(rows)} solved, {wrote} wrote':>26}")
        print(f"{arm:<11}" + "".join(cells))

    # The mechanism P21 actually named: writing to the DB on a task that must not be written to.
    print("\nP21 mechanism — DB writes on tasks whose gold has no write:")
    no_write = {t for t, k in kinds.items() if k in ("refusal", "readonly")}
    for arm in ARMS:
        rows = [r for b in per_arm[arm].values() for r in b if r["task"] in no_write]
        wrote = sum(1 for r in rows if r["wrote"])
        escalated = sum(1 for r in rows if r["escalated"])
        print(
            f"  {arm:<11} wrote {wrote}/{len(rows)}   escalated correctly {escalated}/{len(rows)}"
        )

    # `db_check` compares a hash, so it cannot tell "declined correctly" from "tried to write and
    # the API refused". That distinction is the whole of P21, so it has to be read off the tool
    # responses rather than the reward.
    print("\nHow the solved tasks were actually solved:")
    for arm in ARMS:
        data = json.loads((RESULTS / f"{arm}.json").read_text())
        for sim in data["simulations"]:
            if ((sim.get("reward_info") or {}).get("reward") or 0.0) < 1.0:
                continue
            attempted, errored = 0, 0
            pending: list[str] = []
            for message in sim["messages"]:
                if message.get("role") == "assistant":
                    pending = [c["name"] for c in (message.get("tool_calls") or [])]
                elif message.get("role") == "tool" and pending:
                    name = pending.pop(0)
                    if name in WRITES:
                        attempted += 1
                        errored += str(message.get("content", "")).startswith("Error")
            verdict = (
                f"{errored}/{attempted} write attempts rejected by the API"
                if attempted
                else "made no write attempt"
            )
            print(f"  {arm:<11} task {str(sim['task_id']):<3} {verdict}")


if __name__ == "__main__":
    main()
