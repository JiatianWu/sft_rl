"""Procedural task generation for tau-retail-lite.

A task is a seed plus a family. The sampler builds the database, coerces it so the task
is always feasible, writes the user instruction, and records the oracle action list. The
ground-truth final state is then *derived* by executing that oracle list, so the state
check and the action list can never disagree.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any

from .db import (
    DELIVERED,
    PENDING,
    WRITE_ACTIONS,
    _money,
    apply_action,
    build_db,
    db_hash,
)

FAMILIES = [
    "cancel_order",
    "modify_address",
    "return_items",
    "exchange_items",
    "lookup_and_report",
    "refuse_invalid",
]

# Families ordered by how many tool calls the oracle needs. Used for the RL curriculum.
EASY_FAMILIES = ["lookup_and_report", "cancel_order", "refuse_invalid"]


def _number_variants(value: float) -> list[str]:
    """Acceptable renderings of a monetary amount, so formatting does not decide reward."""
    variants = {f"{value:.2f}", f"{value:g}", f"{value:,.2f}"}
    if float(value).is_integer():
        variants.add(str(int(value)))
    return sorted(variants)


def _order_id_variants(order_id: str) -> list[str]:
    """Acceptable renderings of an order id.

    Money already had this; ids did not, so `"Your order W3006 has been exchanged."` scored
    zero for omitting a `#`. That is a correct answer by any standard a user would apply, and
    it is not a rare edge: an order id is the *entire* required output of both
    `modify_address` and `exchange_items`, a third of the evaluation split.

    Matching stays a substring test, so the bare form is the permissive one and the `#` form
    is redundant — it is kept so the accepted set is readable rather than implied.
    """
    return sorted({order_id, order_id.lstrip("#")})


@dataclass
class TaskSpec:
    family: str
    seed: int
    difficulty: str
    instruction: str
    db: dict[str, Any]
    oracle_actions: list[dict[str, Any]]
    required_outputs: list[list[str]] = field(default_factory=list)
    expected_hash: str = ""

    @property
    def oracle_writes(self) -> list[dict[str, Any]]:
        return [a for a in self.oracle_actions if a["name"] in WRITE_ACTIONS]


def _pick_order(rng: random.Random, db: dict[str, Any], status: str) -> tuple[str, dict[str, Any]]:
    """Pick an order and coerce it to `status` so the family is always feasible."""
    order_id = rng.choice(sorted(db["orders"].keys()))
    order = db["orders"][order_id]
    order["status"] = status
    return order_id, order


def sample_task(seed: int, family: str | None = None, difficulty: str = "easy") -> TaskSpec:
    """Sample a fully specified, verifiable task."""
    rng = random.Random(seed)
    db = build_db(seed)
    family = family or rng.choice(FAMILIES)

    oracle: list[dict[str, Any]] = []
    required: list[list[str]] = []

    if family == "cancel_order":
        order_id, order = _pick_order(rng, db, PENDING)
        user = db["users"][order["user_id"]]
        reason = rng.choice(["no longer needed", "ordered by mistake"])
        oracle = [{"name": "cancel_pending_order", "args": {"order_id": order_id, "reason": reason}}]
        required = [_number_variants(order["total"])]
        instruction = (
            f"Hi, I need to cancel an order. My email is {user['email']}. "
            f"The reason is that it is {reason}. Please tell me the refund amount."
        )
        if difficulty == "easy":
            instruction += f" The order id is {order_id}."

    elif family == "modify_address":
        order_id, order = _pick_order(rng, db, PENDING)
        user = db["users"][order["user_id"]]
        new = {"address1": "742 Evergreen Terrace", "city": "Portland", "state": "OR", "zip": "97201"}
        oracle = [{"name": "modify_pending_order_address", "args": {"order_id": order_id, **new}}]
        required = [_order_id_variants(order_id)]
        instruction = (
            f"Hello, I moved and need the delivery address changed on my pending order. "
            f"My email is {user['email']}. The new address is {new['address1']}, "
            f"{new['city']}, {new['state']} {new['zip']}. Confirm the order id when done."
        )
        if difficulty == "easy":
            instruction += f" The order id is {order_id}."

    elif family == "return_items":
        order_id, order = _pick_order(rng, db, DELIVERED)
        user = db["users"][order["user_id"]]
        item = rng.choice(order["items"])
        oracle = [
            {"name": "return_delivered_order_items", "args": {"order_id": order_id, "item_ids": [item["item_id"]]}}
        ]
        # An order can list the same item_id on more than one line (7.8% of orders do), and the
        # tool refunds *every* matching line. Requiring a single unit price made those tasks
        # unsatisfiable: the agent truthfully reports what the tool returned and is marked wrong.
        refund = _money(sum(i["price"] for i in order["items"] if i["item_id"] == item["item_id"]))
        required = [_number_variants(refund)]
        instruction = (
            f"I want to return the {item['name']} ({item['option']}) from a delivered order. "
            f"My email is {user['email']}. How much will I get back?"
        )
        if difficulty == "easy":
            instruction += f" The order id is {order_id} and the item id is {item['item_id']}."

    elif family == "exchange_items":
        order_id, order = _pick_order(rng, db, DELIVERED)
        user = db["users"][order["user_id"]]
        item = rng.choice(order["items"])
        product = db["products"][item["product_id"]]
        alternatives = [v for v in product["variants"].values() if v["item_id"] != item["item_id"]]
        if not alternatives:  # single-variant product: fall back to a family that always works
            return sample_task(seed + 100_000, "return_items", difficulty)
        new_item = rng.choice(alternatives)
        product["variants"][new_item["item_id"]]["available"] = True
        oracle = [
            {
                "name": "exchange_delivered_order_items",
                "args": {
                    "order_id": order_id,
                    "item_ids": [item["item_id"]],
                    "new_item_ids": [new_item["item_id"]],
                },
            }
        ]
        required = [_order_id_variants(order_id)]
        instruction = (
            f"I'd like to exchange the {item['name']} ({item['option']}) I received for the "
            f"{new_item['option']} version. My email is {user['email']}. Confirm the order id when done."
        )
        if difficulty == "easy":
            instruction += f" The order id is {order_id}, current item id {item['item_id']}, "
            instruction += f"and I want item id {new_item['item_id']}."

    elif family == "lookup_and_report":
        order_id, order = _pick_order(rng, db, rng.choice([PENDING, DELIVERED]))
        user = db["users"][order["user_id"]]
        oracle = []  # read-only: the correct final state is the untouched database
        required = [_number_variants(order["total"])]
        instruction = (
            f"Can you tell me the total amount of one of my orders? My email is {user['email']}. "
            f"Please state the total."
        )
        if difficulty == "easy":
            instruction += f" The order id is {order_id}."

    elif family == "refuse_invalid":
        order_id, order = _pick_order(rng, db, DELIVERED)
        user = db["users"][order["user_id"]]
        # The user asks for something policy forbids: cancelling an already-delivered order.
        oracle = [{"name": "transfer_to_human", "args": {}}]
        required = []
        instruction = (
            f"Please cancel my order {order_id} right now, I don't want it anymore. "
            f"My email is {user['email']}."
        )

    else:
        raise ValueError(f"unknown family: {family}")

    # Derive ground truth by executing the oracle against a copy of the initial database.
    expected_db = copy.deepcopy(db)
    for action in oracle:
        if action["name"] in WRITE_ACTIONS:
            ok, _ = apply_action(expected_db, action["name"], action["args"])
            if not ok:
                raise RuntimeError(f"oracle action failed for seed={seed} family={family}: {action}")

    return TaskSpec(
        family=family,
        seed=seed,
        difficulty=difficulty,
        instruction=instruction,
        db=db,
        oracle_actions=oracle,
        required_outputs=required,
        expected_hash=db_hash(expected_db),
    )
