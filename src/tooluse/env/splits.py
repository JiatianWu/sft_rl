"""Train/test splits, defined once so training and evaluation cannot disagree.

Two separations are enforced:

* **Seeds.** Training and evaluation draw from disjoint seed ranges, and because a seed
  determines the entire database, an evaluation task shares no users, orders or products
  with any training task.
* **A held-out family.** `exchange_items` never appears in RL training. It measures
  whether GRPO improved the tool-use protocol in general or only memorised the families
  it was rewarded on.
"""

from __future__ import annotations

from .tasks import FAMILIES

TRAIN_SEEDS = range(0, 2000)
TEST_SEEDS = range(100_000, 100_100)

HELD_OUT_FAMILY = "exchange_items"
RL_TRAIN_FAMILIES = [f for f in FAMILIES if f != HELD_OUT_FAMILY]
EVAL_FAMILIES = list(FAMILIES)

assert not set(TRAIN_SEEDS) & set(TEST_SEEDS), "train and test seeds must be disjoint"

__all__ = [
    "EVAL_FAMILIES",
    "HELD_OUT_FAMILY",
    "RL_TRAIN_FAMILIES",
    "TEST_SEEDS",
    "TRAIN_SEEDS",
]
