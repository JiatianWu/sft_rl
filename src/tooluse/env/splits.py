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

from .tasks import ABSTAIN_FAMILY, FAMILIES, TEST_SEED_FLOOR

TRAIN_SEEDS = range(0, 2000)
TEST_SEEDS = range(100_000, 100_100)

HELD_OUT_FAMILY = "exchange_items"
RL_TRAIN_FAMILIES = [f for f in FAMILIES if f != HELD_OUT_FAMILY]
EVAL_FAMILIES = list(FAMILIES)

# The abstention family is opt-in on both sides rather than simply appended, because appending it
# would change the meaning of `pass^1` and quietly break comparability with every arm already
# measured on the six-family, 600-task split. Arms that train on it are compared to arms that do
# not *on the original split*, and the new family is reported as its own number.
RL_TRAIN_FAMILIES_WITH_ABSTAIN = RL_TRAIN_FAMILIES + [ABSTAIN_FAMILY]
ABSTAIN_EVAL_FAMILIES = [ABSTAIN_FAMILY]

assert not set(TRAIN_SEEDS) & set(TEST_SEEDS), "train and test seeds must be disjoint"
assert ABSTAIN_FAMILY not in EVAL_FAMILIES, "the headline eval split must stay at six families"
# `tasks.py` picks the abstention topic pool by comparing the seed to this floor, and cannot
# import this module without a cycle. If the ranges ever move, the topic holdout silently stops
# holding anything out, so the two are tied together here.
assert max(TRAIN_SEEDS) < TEST_SEED_FLOOR <= min(TEST_SEEDS), "abstention topic holdout is broken"

__all__ = [
    "ABSTAIN_EVAL_FAMILIES",
    "EVAL_FAMILIES",
    "HELD_OUT_FAMILY",
    "RL_TRAIN_FAMILIES",
    "RL_TRAIN_FAMILIES_WITH_ABSTAIN",
    "TEST_SEEDS",
    "TRAIN_SEEDS",
]
