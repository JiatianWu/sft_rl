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
