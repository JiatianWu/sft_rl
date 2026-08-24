# BFCL pre-registration

Written and committed **before** running BFCL, so the predictions cannot be adjusted to fit
the result. Every claim below is stated with the observation that would refute it, and all
outcomes get reported in `WRITEUP.md` whether they support the existing conclusions or not.

## Why run it at all

The headline number, `0.797` for SFT+RL (30 steps), is measured on an environment I wrote.
Disjoint seeds and a held-out task family rule out memorising *tasks*, but nothing in this
repo shows the gains survive off-distribution. BFCL is the external check: fully offline, no
LLM judge, deterministic AST and state-transition scoring, and zero overlap with
`tau-retail-lite`. It is the one measurement that can distinguish "learned to use tools" from
"learned to do this environment".

## What will be run

- **Categories:** multi-turn, AST / non-live, and relevance-irrelevance, **reported
  separately**. Not BFCL V4 `Overall`, whose weighting is 40% agentic (web search and
  long-session memory) — capabilities this pipeline never trained and that need external APIs.
  Reporting a composite dominated by an untrained skill would be meaningless in both directions.
- **Checkpoints:** `base`, `sft`, `grpo` (SFT+RL 30), `rl_only_long` (RL only 200).
- **Isolation:** a separate Modal image. The training image is version-pinned to a combination
  that took real effort to get working (`vllm 0.27.1`, `transformers 5.15.1`, `trl 1.10.0`);
  `bfcl-eval` will pull its own. The worst outcome is getting no BFCL numbers *and* breaking a
  working pipeline.

## Predictions

**P1 — the load-bearing one. `grpo` will beat `rl_only_long` on BFCL multi-turn, reversing
their in-domain order.**

In `tau-retail-lite`, `rl_only_long` scores 0.929 against `grpo`'s 0.797. §3.3 claims that lead
is a reward hack: it skips user identification in 98.4% of write episodes and fires the write
with the id leaked in the prompt, which works only because my instructions leak ids and my
reward ignores conduct. BFCL leaks nothing and scores conduct, so the shortcut should not
transfer.

*Refuted if* `rl_only_long` scores at or above `grpo` on multi-turn. That would mean it learned
something more general than I claimed and §3.3 is wrong.

**P2 — `sft` will score at or below `base` on the AST and format-sensitivity categories.**

§3.1 found SFT emitting `<function-call>` against a `</tool_call>` close: right tool, right
arguments, wrong wire format. If that corruption is real and general rather than an artifact of
my harness's parser, an external syntax-scored benchmark should see it too.

*Refuted if* `sft` scores clearly above `base` there.

**P3 — absolute scores will be low across the board; multi-turn accuracy under 0.15 for every
arm.** This is a 0.6B model and BFCL multi-turn is hard. Stated so that a low number is not
later dressed up as a surprise.

**P4 — the RL-only arms will show weaker restraint (relevance / irrelevance) than the
SFT-initialised arms**, mirroring `refuse_invalid` collapsing to 0.00 without an SFT prior.

## The failure mode that would make this uninformative

If all four checkpoints score ~0 on multi-turn, P1 is untestable — not refuted, just
unmeasurable, and it will be reported that way rather than as support. This is why a smoke test
on the cheapest category runs first: if the base model cannot register above zero on
single-turn AST, the full sweep buys nothing and should not be paid for.

## What gets reported either way

Per-category numbers for all four arms, the smoke-test result even if it ends the experiment,
and an explicit statement of which predictions survived. A prediction that fails is the most
informative outcome available here, since it would mean a conclusion already in `WRITEUP.md`
needs revising.

---

# Outcome

Added after the run. Full numbers and discussion in `WRITEUP.md` §3.6; reproduce with
`python scripts/bfcl_table.py`.

| prediction | verdict |
|---|---|
| **P1** grpo > rl_only_long on multi-turn | **unresolved** — 14/200 vs 13/200, p=0.84 |
| **P2** sft ≤ base on AST | **confirmed**, far more strongly than predicted |
| **P3** multi-turn under 0.15 for all arms | **confirmed** — 0.020 to 0.080 |
| **P4** RL-only shows weaker restraint than SFT-initialised | **refuted** |

**P1 failed for lack of power, not because the evidence went the other way.** The direction
matches the prediction and the magnitude is a single test case. One multi-turn category at
n=200 and ~6% accuracy carries a ±3.3-point interval, so only a five-point gap would have
registered. Calling this "consistent with the reward-hack account" would be reading noise;
it is reported as unresolved. Resolving it needs all four multi-turn categories (~8 GPU-hours).

**P2 confirmed on a scale I did not anticipate.** I predicted SFT would score "at or below" base
on AST from tag corruption. Actual: pooled AST 0.807 → 0.371, with `parallel` and
`parallel_multiple` at exactly 0.000. The mechanism was not corrupted tags at all — the syntax is
valid and the *count* is wrong, one call emitted where two are required, because every
trajectory it ever trained on had exactly one. A stronger result than predicted, for a different
reason than predicted.

**P4 refuted.** RL-only abstains *better* than SFT+RL (0.738 vs 0.634 on should-not-call), the
opposite of the prediction drawn from `refuse_invalid` collapsing to 0.00 in-domain. In-domain
restraint behaviour did not transfer as an ordering between arms.

**The failure mode named in advance nearly happened, for a different reason.** The pre-registered
worry was all-zero multi-turn scores. Instead the first complete sweep returned four
*statistically identical* arms — because a merge bug had overwritten every checkpoint with base
weights, so BFCL scored the base model four times. That would have produced a clean, quotable,
entirely false "nothing transfers" result. It was caught by hashing the merged weights rather
than by anything in the numbers, which looked perfectly reasonable. The discarded sweep is kept
in `results/bfcl_noise_floor/`: four byte-identical models scored independently, which measures
BFCL's run-to-run spread at 0.012–0.017 and sets the floor any claimed difference must clear.
