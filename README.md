# Multi-turn tool use: SFT → GRPO → eval on a 0.6B model

A minimal but complete pipeline that takes `Qwen3-0.6B` from base weights, through LoRA
instruction tuning on multi-turn tool-use trajectories, through online RL (GRPO) against
an environment with verifiable rewards, and evaluates all three checkpoints identically.

- **[PLAN.md](PLAN.md)** — the plan, written before any code.
- **[WRITEUP.md](WRITEUP.md)** — results, reward design, trade-offs, and what failed.

## The loop

| Stage | What it does | Entry point |
|---|---|---|
| Data | APIGen-MT-5k → chat trajectories with tool calls | `tooluse.data.prepare_sft` |
| SFT | LoRA, assistant-only loss | `tooluse.train.sft` |
| RL | GRPO against `tau-retail-lite` | `tooluse.train.grpo` |
| Eval | held-out tasks, pass^k, decomposed metrics | `tooluse.eval.run_eval` |

## Quick start

```bash
pip install -e .
pytest                                  # environment, reward and template invariants

modal run modal_app.py::prepare         # build the SFT dataset into the volume
modal run modal_app.py::smoke           # verify the stack before spending GPU time
modal run modal_app.py::evaluate --tag base
modal run modal_app.py::sft
modal run modal_app.py::evaluate --tag sft --adapter /work/checkpoints/sft
modal run modal_app.py::grpo
modal run modal_app.py::evaluate --tag grpo --adapter /work/checkpoints/grpo
```

`TOOLUSE_GPU` selects the accelerator (defaults to `A10`, the largest tier reachable on a
Modal account without a payment method on file).

## The environment: `tau-retail-lite`

A stateful retail back office in the spirit of τ-bench's retail domain: a seeded database
of users, orders and products, ten tools, and six task families (`cancel_order`,
`modify_address`, `return_items`, `exchange_items`, `lookup_and_report`,
`refuse_invalid`). Tasks are generated procedurally, so train and test share no database.

Rewards are grounded, never model-judged. Following τ-bench, the outcome term is a
**product**:

```
r_outcome = r_action * r_output
  r_action = final database state hash == ground truth
  r_output = every required fact appears in what the agent told the user
```

The ground-truth state is *derived* by executing the task's oracle action list, so the
oracle and the reward cannot drift apart. `tests/test_env.py` asserts the oracle scores
1.0 on every family; if it ever does not, the task is unsolvable and any RL number on it
would be meaningless.

See [WRITEUP.md](WRITEUP.md) for the shaping terms, how sparse signal is handled, and the
reward-hacking hole the tests caught.

## Layout

```
src/tooluse/
  env/      db.py  tasks.py  retail.py  reward.py  splits.py
  data/     prepare_sft.py  masking.py
  train/    sft.py  grpo.py  rewards.py
  eval/     harness.py  run_eval.py
tests/      env, harness and chat-template invariants
modal_app.py
results/    raw per-episode JSON for every checkpoint
```
