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

# Follow-up: does mixing in parallel-call data repair `parallel`?

Pre-registered before training the `sft_mixed` arm, same rules as above.

§3.6 attributes `parallel` scoring exactly 0/200 to a specific, narrow cause: neither training
stage ever showed the model two calls in one turn (16,732 tool-calling messages in the SFT
corpus, none with more than one). The competing explanation is duller and more worrying — that
LoRA SFT at this scale degrades the model broadly, and parallel calling is simply the most
fragile thing to break first.

Those predict different things, so the corpus was changed and nothing else.

**The intervention.** `sft_mixed` trains on 500 APIGen-MT trajectories plus 500 from
NousResearch/hermes-function-calling-v1, which supplies what APIGen lacks: 56.8% of its
tool-calling turns carry more than one call, across 35 domains. Total trajectories stay at
1,000 to match the existing `sft` arm, so composition is the only variable. xLAM was the first
choice and is gated (`gated=auto`, needs a token); ToolACE is ungated but encodes calls as a
bracketed DSL with spaces in function names, needing a bespoke parser. Hermes needs neither.

**P5 — `sft_mixed` scores materially above zero on `parallel` and `parallel_multiple`.** Any
clearly non-zero result confirms absence-of-examples over general degradation. I expect it to
land below base (0.725/0.750) rather than at it, since only 12.9% of the mixed corpus's
tool-calling turns are multi-call.

*Refuted if* it stays at or near zero, which would mean the damage is broad rather than
specific and §3.6's explanation is wrong.

**P6 — pooled AST recovers well above `sft`'s 0.371**, since two of its four categories were
pinned at zero. Weak, and stated mainly so a rise cannot be presented as a surprise.

**P7 — in-domain `pass^1` stays poor, in the 0.03–0.15 band.** The mix does nothing about the
actual in-domain failure (asking a user simulator that does not exist), and halves the APIGen
data. The real question this leaves open is whether SFT still works as an RL prior, which needs
a GRPO run from `sft_mixed` and is not part of this step.

**What would make this uninformative.** If `sft_mixed` is broadly worse than `sft` everywhere,
the comparison is confounded by having half the APIGen data rather than by the added parallel
examples, and no clean claim is available either way.

---

# Follow-up 2: is the mixed corpus still a good RL prior?

Pre-registered before training `grpo_mixed`, same rules.

SFT's value in this project was never as a policy — it *lowered* in-domain success to 0.035 —
but as an initialisation worth +0.301 (§3.1), and as the thing that anchored the agent to a
procedure the reward never paid for (§3.3, lookup compliance 1.00 where RL-only fell to 0.02).
`sft_mixed` changes the corpus that produced both effects, so both have to be re-measured.
GRPO runs 30 steps from `sft_mixed`, matching the `grpo` arm exactly, so the prior is the only
variable.

There is a real argument in each direction, which is what makes it worth running. Against: half
the domain-matched multi-turn data is gone. For: SFT's in-domain pathology was *asking instead of
acting*, and the mix already moves the model toward acting (should-call 0.375 → 0.812), which is
precisely the failure GRPO otherwise had to spend its budget repairing.

**P8 — `grpo_mixed` beats RL-only at matched compute (0.496).** The weak claim that the mixed
corpus is still worth using as an initialisation at all.

*Refuted if* it lands at or below 0.496, meaning the mix destroyed SFT's value as a prior.

**P9 — `grpo_mixed` shows lower lookup compliance than `grpo`'s 1.00.** §3.3 argues the anchoring
came from SFT's over-caution, and the mix traded exactly that away. If anchoring is a side effect
of over-caution rather than of SFT per se, diluting the over-caution should let RL drift toward
the §3.3 shortcut.

*Refuted if* compliance stays at or near 1.00, which would mean SFT anchors procedure for some
reason other than timidity.

**P10 — `grpo_mixed` keeps parallel calling on BFCL, above 0.4.** RL-only preserved base
capability (§3.6), so RL from a prior that *has* parallel calling should not destroy it, even
though the RL environment is single-call. This is the test of whether the §3.7 repair survives
the second stage — a repair that RL undoes would be worthless in this pipeline.

*Refuted if* `parallel` falls back toward zero, which would mean single-call RL erases it
regardless of the prior and the fix has to be applied after RL, not before.

---

# Follow-up 3: can restraint be trained back without losing the rest?

Pre-registered before training `grpo_abstain`, same rules.

§3.9 found the mechanism behind the collapsing restraint column, and it is not the absence of a
reward. `refuse_invalid` pays full reward for declining, but only via `transfer_to_human`, a write,
so a text-only decline scores exactly 0.0 and successful refusals average 4.5 calls. A fifth of RL
training therefore teaches that an out-of-policy request warrants four to five calls, which is the
opposite of what BFCL irrelevance scores.

**The intervention.** A new family, `irrelevant_request`: the user asks something no tool here can
serve, and the correct episode makes *no call at all* and says so. Restraint is folded into the
state term rather than into progress, so calling anything fails the task outright rather than
merely scoring untidily — reads leave the database untouched, so the hash check alone cannot tell a
refusal from four wasted lookups followed by one. Topics are held out between train and test the
way databases already are, so the in-domain number measures restraint rather than recall of 24
strings. GRPO then runs 30 steps from `sft_mixed`, matching `grpo_mixed` exactly, so the training
mix is the only variable. The system prompt is deliberately untouched, and the headline 600-task
split is unchanged (verified by fingerprint), so every previously reported number stays comparable.

**P11 — BFCL restraint recovers materially above `grpo_mixed`'s 0.460.** This is the whole point:
an in-domain family that is graded on not calling should move the external metric graded on not
calling. I expect it to land between `grpo_mixed` and base's 0.798 rather than above base, since
one family in six is a weak signal against a corpus and a reward that both favour acting.

*Refuted if* restraint stays within noise of 0.460 (the floor is 0.012–0.017), which would mean
in-domain abstention training does not transfer and the act/abstain axis is set by the SFT corpus
rather than by RL.

**P12 — AST holds within noise of `grpo_mixed`'s 0.755.** Restraint and syntax should be separable;
if buying restraint costs parallel calling again, the "one dial" reading is stronger than I think
and the two are not independently addressable at this scale.

**P13 — in-domain `pass^1` on the unchanged six families stays within ±0.05 of 0.475.** One added
family in six dilutes the others by ~17%, so some drop is expected, but a large fall would mean
teaching restraint costs task completion — which is the trade this experiment exists to price.

**P14 — `refuse_invalid` drops.** The new family says "when you cannot help, call nothing"; the old
one says "when the policy forbids it, call `transfer_to_human`". Those are genuinely adjacent and
the model has to separate them on a distinction the system prompt draws only implicitly. This is
the predicted *cost*, stated in advance so a drop cannot later be presented as an acceptable
detail.

**What would make this uninformative.** If the model learns to abstain by never calling anything
anywhere, in-domain `pass^1` will collapse and BFCL restraint will look excellent for the wrong
reason. The should-call column (`live_relevance`) and `pass^1` are the guards; both have to hold up
for a restraint gain to mean anything.

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

## Follow-up outcome: P5–P7

| prediction | verdict |
|---|---|
| **P5** `sft_mixed` materially above zero on parallel | **confirmed** — 0.000 → 0.705 / 0.590 |
| **P6** pooled AST recovers well above 0.371 | **confirmed** — 0.728 |
| **P7** in-domain `pass^1` stays in 0.03–0.15 | **confirmed** — 0.055 |

`parallel` returns to 0.705 against base's 0.725, and pooled AST recovers 82% of what SFT
destroyed, from changing nothing but which trajectories the model read. The narrow explanation in
§3.6 — a missing demonstration rather than lost capacity — is correct.

**One unpredicted cost, and it is not small.** The repair moved the model along the act/abstain
axis: should-not-call falls 0.903 → 0.642 (base 0.801) while should-call rises 0.375 → 0.812. Every
Hermes trajectory calls a function, so a corpus half made of them teaches eagerness. `sft_mixed`
is therefore not strictly better than `sft` — better where the old arm was catastrophic, worse
where it was strong. The confounder named in advance did not materialise: this is a
redistribution along one axis, not uniform degradation from halving the APIGen data.

**The near-miss worth recording.** All 284 multi-call trajectories were silently dropped by the
masker, because Qwen3's template merges consecutive tool messages and breaks the incremental
prefix check. The arm would have trained on zero parallel examples while reporting a full 1,000,
and P5 would have been "refuted" by a bug rather than by evidence. Caught by counting survivors
before spending GPU time; pinned by a test.

## Follow-up 2 outcome: P8–P10

| prediction | verdict |
|---|---|
| **P8** `grpo_mixed` beats RL-only's 0.496 | **refuted** — 0.475, the prior is worth nothing |
| **P9** lookup compliance falls below `grpo`'s 1.00 | **refuted** — 0.998, anchoring fully intact |
| **P10** `parallel` stays above 0.4 through RL | **confirmed** — 0.705, unchanged |

Two of three wrong, and the misses are the informative part.

**P8 — the mixed corpus is worthless as an RL prior.** 30 GRPO steps from `sft_mixed` reach 0.475,
against 0.797 from `sft` and 0.496 from no prior at all. The +0.301 that was the entire case for
running SFT did not survive halving the APIGen data: what remains is indistinguishable from
starting cold, and if anything a shade below it. The per-family breakdown says where it went —
`return_items` 0.16 against `grpo`'s 1.00 and RL-only's 0.91, `cancel_order` 0.23 against 0.66.
The write-heavy families, the ones that need retail protocol rather than general tool syntax, are
exactly what the removed 500 trajectories carried.

**P9 — anchoring is not a side effect of timidity.** P9's reasoning was that §3.3's lookup
compliance came from SFT's over-caution, so a corpus that made the model eager should let RL drift
toward the shortcut. It did not: compliance is 0.998, statistically identical to `grpo`'s 1.000,
in the *most* eager checkpoint in the project (should-not-call 0.454). Eagerness and
procedure-following are independent axes. Whatever SFT installs that resists the §3.3 reward hack
is carried by both corpora and is not the same thing as reluctance — better news than P9 being
right, since it means the anchoring is robust to changing the data.

**P10 — the repair survives RL.** `parallel` is 0.705 before and after, `parallel_multiple` goes
0.590 → 0.605, pooled AST 0.728 → 0.755. Thirty steps of single-call RL do not erase parallel
calling once the prior has it, so §3.7's fix belongs before RL and stays there. Restraint degrades
further and by a lot, though: pooled 0.645 → 0.460, should-not-call 0.642 → 0.454, the worst of any
checkpoint measured. Both stages push the same way on the act/abstain axis.

My first reading of that — nothing in the pipeline pays for abstaining — was wrong, and checking it
against the environment source rather than against my own summary of it gave a better answer
(§3.9). `refuse_invalid` pays full reward for declining, but only through `transfer_to_human`,
which is a write, so a text-only decline scores exactly 0.0 and successful refusals average 4.5
calls. A fifth of RL training therefore teaches that out-of-policy requests warrant four to five
calls — the opposite of what BFCL irrelevance scores. And that requirement is itself a patch for an
earlier hack in which doing nothing scored a perfect 1.0.

**The failure mode named in advance nearly happened, for a different reason.** The pre-registered
worry was all-zero multi-turn scores. Instead the first complete sweep returned four
*statistically identical* arms — because a merge bug had overwritten every checkpoint with base
weights, so BFCL scored the base model four times. That would have produced a clean, quotable,
entirely false "nothing transfers" result. It was caught by hashing the merged weights rather
than by anything in the numbers, which looked perfectly reasonable. The discarded sweep is kept
in `results/bfcl_noise_floor/`: four byte-identical models scored independently, which measures
BFCL's run-to-run spread at 0.012–0.017 and sets the floor any claimed difference must clear.
