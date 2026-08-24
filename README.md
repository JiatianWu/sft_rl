# Multi-turn tool use: SFT → GRPO → eval on a 0.6B model

A minimal but complete pipeline that takes `Qwen3-0.6B` from base weights, through LoRA
instruction tuning on multi-turn tool-use trajectories, through online RL (GRPO) against
an environment with verifiable rewards, and evaluates all three checkpoints identically.

- **[PLAN.md](PLAN.md)** — the plan, written before any code.
- **[WRITEUP.md](WRITEUP.md)** — results, reward design, trade-offs, and what failed.
- **[BFCL_PREREGISTRATION.md](BFCL_PREREGISTRATION.md)** — external-benchmark predictions,
  committed before the run, with the outcome appended.
- **[ENGINEERING_NOTES.md](ENGINEERING_NOTES.md)** — the things that cost time.

## Results

Task success on 2,400 held-out episodes per checkpoint, identical decoding throughout.
"30"/"200" are GRPO steps; RL-only arms train a fresh LoRA from the same config SFT uses.

| | Base | + SFT | RL only (30) | SFT+RL (30) | RL only (200) | SFT+RL (200) |
|---|---|---|---|---|---|---|
| **pass^1** | 0.132 | 0.035 | 0.496 | **0.797** | *0.929* | 0.868 |
| **looked up before writing** | 0.14 | 0.42 | 0.94 | **1.00** | **0.02** | **1.00** |
| illegal writes / episode | 0.047 | 0.000 | 0.186 | 0.169 | 0.240 | 0.193 |

Three findings, none visible from the headline number alone:

- **SFT hurt** (0.132 → 0.035) — APIGen-MT teaches it to ask a user simulator for missing
  information, and this environment has none, so half of SFT episodes make no tool call.
- **SFT is still worth +0.301 as an RL prior.** At matched compute RL reaches 0.496 from
  scratch and 0.797 from the SFT adapter. Worthless as a policy, decisive as an initialisation.
- **The top score, 0.929, is a reward hack.** RL-only at 200 steps skips user identification
  in 98.4% of write episodes and fires the write with the id leaked in the prompt. The best
  genuine agent is **SFT+RL at 30 steps, 0.797**.

A fourth finding came out of the tables rather than the models: `return_items` read exactly
0.93 in three independently trained arms, which turned out to be a benchmark bug making 1.2%
of episodes unwinnable — and the test meant to prevent exactly that was circular. Both are
fixed and pinned, and all six arms were then re-evaluated on the corrected metric.
Re-evaluating twice also gives real error bars: **±0.01 headline, ±0.05 per family**.
[WRITEUP.md](WRITEUP.md) §3.4.

### None of it transfers — BFCL

The table above is measured on an environment I wrote. On BFCL (2,340 test cases, external,
offline, deterministic; predictions committed beforehand in
**[BFCL_PREREGISTRATION.md](BFCL_PREREGISTRATION.md)**):

| | Base | + SFT | SFT+RL (30) | RL only (200) |
|---|---|---|---|---|
| **AST pooled** (1,000) | **0.807** | 0.371 | 0.493 | 0.795 |
| `parallel` (200) | 0.725 | **0.000** | **0.000** | 0.715 |
| `multi_turn_base` (200) | 0.080 | 0.020 | 0.070 | 0.065 |

**No trained arm beats base on anything, and SFT is catastrophic.** Parallel function calling
falls to *exactly zero*: the syntax stays valid, but the model emits one tool call where two are
required, because both the SFT corpus and the environment are one-call-per-turn. A capability
the base model had was trained out of it, and no in-domain metric could see it —
`tau-retail-lite` cannot even express the failing task.

**On-policy RL preserved what imitation destroyed** (0.795 vs base 0.807), despite training in
the same single-call environment. The 0.797 is a real gain in the environment it was trained
for, bought with general capability. [WRITEUP.md](WRITEUP.md) §3.6.

### The diagnosis was actionable

Mixing parallel-call data into SFT — 500 Hermes trajectories replacing 500 APIGen ones, total
held at 1,000 — repairs it, which confirms the cause was a missing demonstration rather than
lost capacity:

| | Base | + SFT | **SFT mixed** |
|---|---|---|---|
| `parallel` | 0.725 | **0.000** | **0.705** |
| **AST pooled** | 0.807 | 0.371 | **0.728** |

82% of the damage recovered by changing nothing but which trajectories the model read. It is not
free: the same change moves the model along the act/abstain axis, trading abstention (0.903 →
0.642) for willingness to call (0.375 → 0.812). [WRITEUP.md](WRITEUP.md) §3.7.

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

The external benchmark runs on a separate image, since `bfcl-eval` pins its own `vllm` and
`transformers`. BFCL ignores LoRA adapters at request time, so checkpoints must be merged into
full weights first — and `verify_merged_differ` is not optional, because a silent merge failure
scores the base model N times and looks like a clean null result:

```bash
modal run modal_app.py::merge_adapters       # LoRA -> standalone models
modal run modal_app.py::verify_merged_differ # assert the arms are actually different models
modal run modal_app.py::bfcl_sweep --jobs "base:simple_python;sft:simple_python"
python scripts/bfcl_table.py                 # per-category table, CIs, significance
```

`bfcl_sweep` fans the arms out across parallel containers, one GPU each: identical GPU-seconds,
wall clock divided by the number of arms.

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
