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

# Deliberately *not* in FAMILIES, so `EVAL_FAMILIES` keeps producing the same 600-task split
# every committed number was measured on. Adding it there would silently make `pass^1`
# incomparable with every arm already reported.
ABSTAIN_FAMILY = "irrelevant_request"
ALL_FAMILIES = FAMILIES + [ABSTAIN_FAMILY]

# Requests no tool in this environment can serve. The correct episode makes *no call at all* and
# says so, which is the one behaviour `tau-retail-lite` could not previously express: every other
# family, `refuse_invalid` included, is only scoreable by leaving a state footprint (§3.9).
#
# Deliberately many, and crossed with openers below. A short list would be trained on ~125 times
# each over a 1,000-task RL run, and the model could reach a perfect in-domain score by memorising
# the strings while learning nothing about restraint — which would look exactly like success and
# would not move BFCL irrelevance at all.
OUT_OF_SCOPE_REQUESTS = [
    "what the weather in Portland is going to be tomorrow",
    "booking me a flight to Denver next Friday",
    "what Apple's share price is right now",
    "a good recipe for risotto",
    "translating 'thank you for your help' into Japanese",
    "reserving a table for four at an Italian place tonight",
    "what 17 percent of 4,830 comes to",
    "who won the football game last night",
    "what the capital of Australia is",
    "setting a reminder for my dentist appointment",
    "writing a birthday message for my sister",
    "what time the pharmacy on Third Street closes",
    "whether this rash on my arm looks serious",
    "how to fix a leaking kitchen tap",
    "what the traffic is like on the freeway right now",
    "summarising the news from this morning",
]

# Held out the way databases are: a test episode's topic never appeared in training. Without this
# the two pools are identical, every test surface form is one the model was trained on, and the
# in-domain abstention score measures memorisation of 24 strings. The external check (BFCL
# irrelevance) is unaffected either way, but an in-domain number that cannot distinguish recall
# from restraint is worse than no number.
HELD_OUT_SCOPE_REQUESTS = [
    "sending an email to my landlord",
    "what my horoscope says today",
    "how many kilometres it is from Boston to Chicago",
    "explaining how a mortgage rate is calculated",
    "which laptop I should buy for video editing",
    "the score of yesterday's cricket match",
    "calling me a taxi to the airport",
    "what the exchange rate is between euros and yen",
]

# Mirrors the seed split in `splits.py`, which cannot be imported here without a cycle. The
# assertion tying the two together lives there.
TEST_SEED_FLOOR = 100_000

# Crossed with the requests above, so ~200 surface forms rather than a couple of dozen.
OUT_OF_SCOPE_OPENERS = [
    "Hi, quick unrelated question — can you help me with {}?",
    "While I have you, could you help with {}?",
    "Hey, I need help with {}.",
    "One more thing: {}, please.",
    "Sorry, off topic, but I was wondering about {}.",
    "Can you help me with {}?",
    "Before I forget — {}?",
    "I'd really appreciate help with {}.",
]

# Any one of these is accepted as having declined. Substring matching is permissive on purpose:
# the graded behaviour is *not calling anything*, and the text check exists only to separate a
# deliberate decline from a model that emitted nothing at all — which is the exact hole that made
# passivity score 1.0 on `refuse_invalid` before escalation was given a state footprint.
DECLINE_PHRASES = [
    "cannot", "can't", "can not", "unable", "not able", "do not have", "don't have",
    "no tool", "not something", "outside", "beyond", "afraid", "only help", "only assist",
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
    # When set, a correct episode makes no tool call whatsoever. Distinct from an empty
    # `oracle_actions`, which only requires refraining from *writes* — `lookup_and_report` has an
    # empty oracle and still expects the model to read.
    expect_no_calls: bool = False

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

    elif family == ABSTAIN_FAMILY:
        # No user email, no order id: nothing here is even addressable by a tool, so a lookup is
        # not merely wasteful but impossible to ground. Scored as BFCL scores irrelevance — any
        # call at all is wrong, including `transfer_to_human`. That is stricter than the system
        # prompt's escalation clause, which covers policy-forbidden *actions* rather than
        # out-of-scope requests; the prompt is left untouched so every other arm stays comparable.
        pool = HELD_OUT_SCOPE_REQUESTS if seed >= TEST_SEED_FLOOR else OUT_OF_SCOPE_REQUESTS
        instruction = rng.choice(OUT_OF_SCOPE_OPENERS).format(rng.choice(pool))
        oracle = []
        required = [list(DECLINE_PHRASES)]

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
        expect_no_calls=family == ABSTAIN_FAMILY,
    )
