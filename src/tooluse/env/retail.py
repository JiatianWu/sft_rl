"""tau-retail-lite: a stateful multi-turn retail environment for GRPO and evaluation.

Contract imposed by TRL's `environment_factory` (verified against trl/trainer/grpo_trainer.py):

* the factory is probed with no arguments, so `__init__` takes none;
* every public method becomes a tool, so all helpers are underscore-prefixed;
* `reset` is called with the whole dataset row as keyword arguments, and its return value
  is *appended* to the last prompt message rather than replacing it;
* instances are pooled and reused across batches, so `reset` must fully reinitialise state.

The environment deliberately does not define `get_reward`. The output half of the reward
needs the assistant's text, which the environment never sees; scoring therefore lives in
trainer-owned reward functions that receive both the completions and these instances.
"""

from __future__ import annotations

import json
from typing import Any

from .db import DELIVERED, PENDING, WRITE_ACTIONS, apply_action
from .reward import Rollout
from .tasks import TaskSpec, sample_task

SYSTEM_PROMPT = """You are a retail customer service agent.

Policy:
- Identify the user before acting. Look them up by the email they give you.
- Only pending orders can be cancelled or have their address modified.
- Only delivered orders can be returned or exchanged.
- If the user asks for something the policy forbids, do not perform the action. Explain why and call transfer_to_human.
- Always tell the user the concrete result, including any amount of money, in your final message.
- Make one tool call at a time, then use its result to decide the next step."""


class RetailEnv:
    """A seeded retail back office exposed as tools."""

    def __init__(self) -> None:
        # Start in a valid state. TRL probes a freshly constructed instance with
        # `inspect.getmembers`, which evaluates every attribute including properties, so
        # an environment that is unusable before `reset()` raises during trainer setup.
        self._spec: TaskSpec
        self._db: dict[str, Any]
        self._rollout: Rollout
        self.reset()

    # ------------------------------------------------------------------ lifecycle

    def reset(self, seed: int = 0, family: str | None = None, difficulty: str = "easy", **kwargs: Any) -> str:
        """Start a new episode and return the user's opening message."""
        self._spec = sample_task(int(seed), family, difficulty)
        self._db = self._spec.db
        self._rollout = Rollout()
        return self._spec.instruction

    # ------------------------------------------------------- internal accessors

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    def _record(self, name: str, args: dict[str, Any], ok: bool, illegal_write: bool = False) -> None:
        self._rollout.calls.append({"name": name, "args": args})
        if not ok:
            self._rollout.failed_calls += 1
        if illegal_write:
            self._rollout.illegal_writes += 1

    def _write(self, name: str, args: dict[str, Any]) -> str:
        ok, message = apply_action(self._db, name, args)
        # A write that the environment rejected is a policy violation, not just a miss.
        self._record(name, args, ok, illegal_write=not ok)
        return message

    # ----------------------------------------------------------------- read tools

    def find_user_id_by_email(self, email: str) -> str:
        """Find a user's id from their email address.

        Args:
            email: The email address of the user, such as 'mei.tanaka12@example.com'.
        """
        for user in self._db["users"].values():
            if user["email"].lower() == str(email).strip().lower():
                self._record("find_user_id_by_email", {"email": email}, True)
                return json.dumps({"user_id": user["user_id"]})
        self._record("find_user_id_by_email", {"email": email}, False)
        return "Error: no user found with that email."

    def get_user_details(self, user_id: str) -> str:
        """Get a user's profile, including their address, payment methods and order ids.

        Args:
            user_id: The id of the user, such as 'mei_tanaka_412'.
        """
        user = self._db["users"].get(user_id)
        self._record("get_user_details", {"user_id": user_id}, user is not None)
        if user is None:
            return f"Error: user {user_id} not found."
        return json.dumps(user)

    def list_user_orders(self, user_id: str) -> str:
        """List the ids, statuses and totals of every order belonging to a user.

        Args:
            user_id: The id of the user, such as 'mei_tanaka_412'.
        """
        user = self._db["users"].get(user_id)
        self._record("list_user_orders", {"user_id": user_id}, user is not None)
        if user is None:
            return f"Error: user {user_id} not found."
        summary = [
            {
                "order_id": oid,
                "status": self._db["orders"][oid]["status"],
                "total": self._db["orders"][oid]["total"],
            }
            for oid in user["orders"]
        ]
        return json.dumps(summary)

    def get_order_details(self, order_id: str) -> str:
        """Get the full details of an order, including its items, status and total.

        Args:
            order_id: The id of the order, such as '#W3004'.
        """
        order = self._db["orders"].get(order_id)
        self._record("get_order_details", {"order_id": order_id}, order is not None)
        if order is None:
            return f"Error: order {order_id} not found."
        return json.dumps(order)

    def get_product_details(self, product_id: str) -> str:
        """Get a product and all of its purchasable variants.

        Args:
            product_id: The id of the product, such as 'P1000'.
        """
        product = self._db["products"].get(product_id)
        self._record("get_product_details", {"product_id": product_id}, product is not None)
        if product is None:
            return f"Error: product {product_id} not found."
        return json.dumps(product)

    # ---------------------------------------------------------------- write tools

    def cancel_pending_order(self, order_id: str, reason: str) -> str:
        """Cancel a pending order and refund it. Only works while the order is pending.

        Args:
            order_id: The id of the order to cancel, such as '#W3004'.
            reason: Why the user wants to cancel the order.
        """
        return self._write("cancel_pending_order", {"order_id": order_id, "reason": reason})

    def modify_pending_order_address(self, order_id: str, address1: str, city: str, state: str, zip: str) -> str:
        """Change the delivery address of a pending order.

        Args:
            order_id: The id of the order to modify, such as '#W3004'.
            address1: The new street address, such as '742 Evergreen Terrace'.
            city: The new city, such as 'Portland'.
            state: The new two-letter state code, such as 'OR'.
            zip: The new postal code, such as '97201'.
        """
        return self._write(
            "modify_pending_order_address",
            {"order_id": order_id, "address1": address1, "city": city, "state": state, "zip": zip},
        )

    def return_delivered_order_items(self, order_id: str, item_ids: list[str]) -> str:
        """Return some items from a delivered order and refund them.

        Args:
            order_id: The id of the delivered order, such as '#W3004'.
            item_ids: The ids of the items to return, such as ['P1000-0'].
        """
        return self._write("return_delivered_order_items", {"order_id": order_id, "item_ids": item_ids})

    def exchange_delivered_order_items(self, order_id: str, item_ids: list[str], new_item_ids: list[str]) -> str:
        """Exchange items from a delivered order for different variants of the same products.

        Args:
            order_id: The id of the delivered order, such as '#W3004'.
            item_ids: The ids of the items being returned, such as ['P1000-0'].
            new_item_ids: The ids of the replacement items, in the same order, such as ['P1000-1'].
        """
        return self._write(
            "exchange_delivered_order_items",
            {"order_id": order_id, "item_ids": item_ids, "new_item_ids": new_item_ids},
        )

    def transfer_to_human(self, summary: str) -> str:
        """Hand the conversation to a human agent when the policy does not allow the request.

        Args:
            summary: A short summary of the user's request and why it cannot be handled.
        """
        return self._write("transfer_to_human", {})


def build_prompt() -> list[dict[str, str]]:
    """The prompt scaffold. `reset()` appends the user instruction to the empty user turn."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ""}]


__all__ = [
    "RetailEnv",
    "SYSTEM_PROMPT",
    "build_prompt",
    "DELIVERED",
    "PENDING",
    "WRITE_ACTIONS",
]
