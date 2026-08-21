"""Seeded retail database plus the pure state transitions the tools wrap.

The database is regenerated from scratch for every rollout so that a task is fully
described by its seed. Keeping the mutations here (rather than inside the environment
class) lets the task sampler execute an oracle action list against a throwaway copy to
derive the ground-truth final state, which guarantees the oracle and the reward can
never drift apart.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from typing import Any

# Small fixed pools. The database is intentionally tiny: it is rebuilt per rollout and
# every tool result lands in a 0.6B model's context window.
FIRST_NAMES = ["yusuf", "mei", "omar", "sofia", "raj", "ingrid", "liam", "noor", "diego", "hana"]
LAST_NAMES = ["rossi", "okafor", "tanaka", "silva", "novak", "khan", "muller", "dubois"]
CITIES = [
    ("San Francisco", "CA", "94102"),
    ("Denver", "CO", "80202"),
    ("Austin", "TX", "73301"),
    ("Boston", "MA", "02108"),
    ("Seattle", "WA", "98101"),
]
STREETS = ["Elm Street", "Oak Avenue", "Maple Drive", "Pine Road", "Cedar Lane"]

PRODUCT_CATALOG = [
    ("Wireless Earbuds", [("black", 89.99), ("white", 89.99), ("blue", 94.50)]),
    ("Coffee Grinder", [("manual", 42.00), ("electric", 78.25)]),
    ("Desk Lamp", [("silver", 35.75), ("black", 35.75)]),
    ("Running Shoes", [("size 9", 120.00), ("size 10", 120.00), ("size 11", 125.00)]),
    ("Water Bottle", [("500ml", 18.40), ("1L", 24.90)]),
    ("Backpack", [("grey", 65.00), ("navy", 68.00)]),
    ("Mechanical Keyboard", [("brown switch", 110.00), ("red switch", 110.00)]),
    ("Yoga Mat", [("green", 29.99), ("purple", 32.50)]),
]

# Order status vocabulary. Which writes are legal depends entirely on these.
PENDING = "pending"
DELIVERED = "delivered"
CANCELLED = "cancelled"
RETURN_REQUESTED = "return requested"
EXCHANGE_REQUESTED = "exchange requested"

# Actions that change state. `transfer_to_human` is included deliberately: escalation is
# a real, recorded act, and without a state footprint a refusal task would reward a model
# that simply says "I can't help" and does nothing.
WRITE_ACTIONS = {
    "cancel_pending_order",
    "modify_pending_order_address",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
    "transfer_to_human",
}


def _money(x: float) -> float:
    return round(float(x) + 1e-9, 2)


def build_db(seed: int) -> dict[str, Any]:
    """Build a deterministic retail database for the given seed."""
    rng = random.Random(seed)

    products: dict[str, Any] = {}
    for p_idx, (name, variants) in enumerate(PRODUCT_CATALOG):
        product_id = f"P{1000 + p_idx}"
        items = {}
        for v_idx, (option, price) in enumerate(variants):
            item_id = f"{product_id}-{v_idx}"
            items[item_id] = {
                "item_id": item_id,
                "product_id": product_id,
                "name": name,
                "option": option,
                "price": _money(price),
                "available": rng.random() > 0.2,
            }
        products[product_id] = {"product_id": product_id, "name": name, "variants": items}

    all_items = [item for p in products.values() for item in p["variants"].values()]

    users: dict[str, Any] = {}
    orders: dict[str, Any] = {}
    order_counter = 3000

    n_users = rng.randint(4, 6)
    used_names: set[str] = set()
    for _ in range(n_users):
        while True:
            first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
            user_id = f"{first}_{last}_{rng.randint(100, 999)}"
            if user_id not in used_names:
                used_names.add(user_id)
                break
        city, state, zipcode = rng.choice(CITIES)
        address = {
            "address1": f"{rng.randint(100, 999)} {rng.choice(STREETS)}",
            "city": city,
            "state": state,
            "zip": zipcode,
        }
        payment_methods = {
            f"credit_card_{rng.randint(1000, 9999)}": {"source": "credit_card", "brand": "visa"},
            f"gift_card_{rng.randint(1000, 9999)}": {"source": "gift_card", "balance": _money(rng.uniform(20, 300))},
        }
        users[user_id] = {
            "user_id": user_id,
            "name": {"first_name": first.capitalize(), "last_name": last.capitalize()},
            "email": f"{first}.{last}{rng.randint(10, 99)}@example.com",
            "address": address,
            "payment_methods": payment_methods,
            "orders": [],
        }

        for _ in range(rng.randint(1, 3)):
            order_counter += 1
            order_id = f"#W{order_counter}"
            n_items = rng.randint(1, 3)
            items = [copy.deepcopy(rng.choice(all_items)) for _ in range(n_items)]
            for item in items:
                item.pop("available", None)
            total = _money(sum(i["price"] for i in items))
            status = rng.choice([PENDING, DELIVERED, DELIVERED])
            orders[order_id] = {
                "order_id": order_id,
                "user_id": user_id,
                "status": status,
                "address": copy.deepcopy(address),
                "items": items,
                "total": total,
                "payment_method_id": rng.choice(list(payment_methods.keys())),
            }
            users[user_id]["orders"].append(order_id)

    # `escalated` is part of the hashed state, so escalating (or failing to) is scored.
    return {"users": users, "products": products, "orders": orders, "escalated": False}


def db_hash(db: dict[str, Any]) -> str:
    """Canonical hash of the database, used as the ground-truth state check."""
    return hashlib.sha256(json.dumps(db, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def apply_action(db: dict[str, Any], name: str, args: dict[str, Any]) -> tuple[bool, str]:
    """Apply a write action to `db` in place.

    Returns (ok, message). A rejected action leaves the database untouched, which is what
    makes an illegal write distinguishable from a legal one at scoring time.
    """
    try:
        if name == "transfer_to_human":
            # Only the fact of escalation is recorded. The agent's free-text summary is
            # excluded from state so that wording cannot decide the state check.
            db["escalated"] = True
            return True, json.dumps({"status": "transferred to a human agent"})

        if name == "cancel_pending_order":
            order = db["orders"].get(args["order_id"])
            if order is None:
                return False, f"Error: order {args['order_id']} not found."
            if order["status"] != PENDING:
                return False, f"Error: order status is '{order['status']}', only pending orders can be cancelled."
            order["status"] = CANCELLED
            order["cancel_reason"] = args.get("reason", "")
            return True, json.dumps({"order_id": order["order_id"], "status": CANCELLED, "refund": order["total"]})

        if name == "modify_pending_order_address":
            order = db["orders"].get(args["order_id"])
            if order is None:
                return False, f"Error: order {args['order_id']} not found."
            if order["status"] != PENDING:
                return False, f"Error: order status is '{order['status']}', address can only be changed while pending."
            order["address"] = {
                "address1": args["address1"],
                "city": args["city"],
                "state": args["state"],
                "zip": args["zip"],
            }
            return True, json.dumps({"order_id": order["order_id"], "address": order["address"]})

        if name == "return_delivered_order_items":
            order = db["orders"].get(args["order_id"])
            if order is None:
                return False, f"Error: order {args['order_id']} not found."
            if order["status"] != DELIVERED:
                return False, f"Error: order status is '{order['status']}', only delivered orders can be returned."
            item_ids = list(args["item_ids"])
            owned = [i["item_id"] for i in order["items"]]
            for item_id in item_ids:
                if item_id not in owned:
                    return False, f"Error: item {item_id} is not part of order {order['order_id']}."
            order["status"] = RETURN_REQUESTED
            order["return_item_ids"] = sorted(item_ids)
            refund = _money(sum(i["price"] for i in order["items"] if i["item_id"] in item_ids))
            order["refund"] = refund
            return True, json.dumps({"order_id": order["order_id"], "status": RETURN_REQUESTED, "refund": refund})

        if name == "exchange_delivered_order_items":
            order = db["orders"].get(args["order_id"])
            if order is None:
                return False, f"Error: order {args['order_id']} not found."
            if order["status"] != DELIVERED:
                return False, f"Error: order status is '{order['status']}', only delivered orders can be exchanged."
            item_ids = list(args["item_ids"])
            new_item_ids = list(args["new_item_ids"])
            if len(item_ids) != len(new_item_ids):
                return False, "Error: item_ids and new_item_ids must have the same length."
            owned = [i["item_id"] for i in order["items"]]
            for item_id in item_ids:
                if item_id not in owned:
                    return False, f"Error: item {item_id} is not part of order {order['order_id']}."
            for new_id in new_item_ids:
                product_id = new_id.rsplit("-", 1)[0]
                product = db["products"].get(product_id)
                if product is None or new_id not in product["variants"]:
                    return False, f"Error: item {new_id} does not exist."
                if not product["variants"][new_id]["available"]:
                    return False, f"Error: item {new_id} is not available."
            order["status"] = EXCHANGE_REQUESTED
            order["exchange_item_ids"] = sorted(item_ids)
            order["exchange_new_item_ids"] = sorted(new_item_ids)
            return True, json.dumps({"order_id": order["order_id"], "status": EXCHANGE_REQUESTED})

    except (KeyError, TypeError) as exc:
        return False, f"Error: malformed arguments for {name}: {exc}"

    return False, f"Error: unknown action {name}."
