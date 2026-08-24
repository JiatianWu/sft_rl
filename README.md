# Multi-turn tool use: SFT → GRPO → eval on a 0.6B model

A minimal but complete pipeline that takes `Qwen3-0.6B` from base weights, through LoRA
instruction tuning on multi-turn tool-use trajectories, through online RL (GRPO) against
an environment with verifiable rewards, and evaluates all three checkpoints identically.

- **[PLAN.md](PLAN.md)** — the plan, written before any code.
- **[WRITEUP.md](WRITEUP.md)** — results, reward design, trade-offs, and what failed.
- **[ENGINEERING_NOTES.md](ENGINEERING_NOTES.md)** — the things that cost time.

## Results

Task success on 2,400 held-out episodes per checkpoint, identical decoding throughout.
"30"/"200" are GRPO steps; RL-only arms train a fresh LoRA from the same config SFT uses.

| | Base | + SFT | RL only (30) | SFT+RL (30) | RL only (200) | SFT+RL (200) |
|---|---|---|---|---|---|---|
| **pass^1** | 0.140 | 0.037 | 0.489 | **0.794** | *0.935* | 0.863 |
| **looked up before writing** | 0.13 | 0.42 | 0.94 | **1.00** | **0.02** | **1.00** |
| illegal writes / episode | 0.050 | 0.002 | 0.185 | 0.171 | 0.221 | 0.189 |

Three findings, none visible from the headline number alone:

- **SFT hurt** (0.140 → 0.037) — APIGen-MT teaches it to ask a user simulator for missing
  information, and this environment has none, so half of SFT episodes make no tool call.
- **SFT is still worth +0.305 as an RL prior.** At matched compute RL reaches 0.489 from
  scratch and 0.794 from the SFT adapter. Worthless as a policy, decisive as an initialisation.
- **The top score, 0.935, is a reward hack.** RL-only at 200 steps skips user identification
  in 98.2% of write episodes and fires the write with the id leaked in the prompt. The best
  genuine agent is **SFT+RL at 30 steps, 0.794**.

A fourth finding came out of the tables rather than the models: `return_items` reads exactly
0.93 in three independently trained arms, which turned out to be a benchmark bug making 1.2%
of episodes unwinnable — and the test meant to prevent exactly that was circular. Fixed and
pinned; reported numbers predate the fix. [WRITEUP.md](WRITEUP.md) §3.4.

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
pytest                            # environment, reward and template invariants
modal run modal_app.py            # the whole loop, from an empty workspace
```

`modal_app.py::main` runs data prep and a smoke test on CPU, evaluates the base checkpoint,
then hands off to `finish`, which does **SFT → eval → GRPO → eval inside one container**.
Keeping the four GPU stages together matters: run as separate Modal functions they cost four
cold starts and re-download the base weights each time, which is roughly fifteen minutes of
paid GPU on nothing. `finish` also commits the volume after every stage, so an interruption
costs one stage rather than the whole run.

```bash
modal run modal_app.py::resume    # skip the base eval when results/ already has it
modal run modal_app.py::smoke     # verify the stack before spending GPU time
```

Individual stages (`prepare`, `sft`, `grpo`, `evaluate`) remain callable for debugging.
`TOOLUSE_GPU` selects the accelerator, defaulting to `A10` — the largest tier reachable on a
Modal account with no payment method on file.

Then pull the results down and render the tables:

```bash
modal volume get tooluse-workspace results ./results --force
python scripts/collect_results.py     # headline + per-family comparison
python scripts/error_taxonomy.py      # what the failures actually are
```

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
