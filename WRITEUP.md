# Multi-turn tool use on a 0.6B model: SFT → GRPO → eval

**Model** `Qwen3-0.6B` · **Compute** Modal, single A10 · **Cost** $32.10 metered, $1.94 billed
**Loop** closed inside the 4-hour cap; the rounds after it are logged in §6 of
[FINDINGS.md](FINDINGS.md). Full detail, every number and every failed prediction live there — this
is the 3-page version.

## 1. The result

Six checkpoints, identical decoding (T=0.7, top-p 0.95, fixed seed), **2,400 held-out episodes
each** (600 tasks × 4 trials). Test seeds are disjoint from training seeds. Error bars are
*measured*, by evaluating every checkpoint twice: **±0.01 headline, ±0.05 per family.**

| | Base | + SFT | RL only (30) | SFT+RL (30) | RL only (200) | **SFT-1500+RL** |
|---|---|---|---|---|---|---|
| **pass^1** | 0.132 | 0.035 | 0.496 | 0.797 | *0.929* | **0.880** |
| looked up before writing | 0.14 | 0.42 | 0.94 | 1.00 | **0.02** | **1.00** |

The loop works: 0.132 → 0.880 in its own environment, a 6.7× gain. **Every other section of this
document is about why that sentence is less impressive than it sounds**, which is the part of the
exercise I found worth the time.

## 2. Environment and reward

`tau-retail-lite` is a τ-bench-style retail environment written for this project: 8 tools over a
JSON database, 6 task families, procedurally generated so train and test are disjoint by seed. Built
rather than adopted because an off-the-shelf benchmark gives comparability but not a *reward*, and
GRPO needs a verifiable signal on every rollout.

**Grounding.** The outcome term follows τ-bench: a state check **times** an output check —
`r_outcome = r_action * r_output`, where `r_action` is `hash(db) == expected` and `r_output` is
whether the required facts appear in the reply. A product, not a sum, so the agent can neither talk
its way to a reward without acting nor act correctly and fail to report back. Hashing the whole DB
makes the check independent of *how* the agent got there.

**Sparse signal.** Outcome alone is near-binary and rarely fires early in training, so three shaping
terms densify it, each shaped so it cannot be farmed on its own:

- **Progress** (+0.3) — fraction of oracle actions performed with matching key arguments, *consumed
  greedily*, so repeating one correct call cannot substitute for a missing one.
- **Efficiency** (−0.05/redundant call, capped −0.15) — **charged only when the episode already
  succeeded**, so it can never outrank correctness.
- **Violations** (−0.2 per illegal write or failed call, capped −0.5) — always charged.

**Abstention is folded into the state term, not the shaping.** For a task whose correct answer is to
decline, "untouched DB" is satisfied by any number of reads, so the hash alone cannot tell a correct
refusal from four wasted lookups. `r_action` therefore also requires that no calls were made, which
lets restraint *fail* an episode rather than merely make it untidy.

## 3. Trade-offs

**Qwen3-0.6B over Gemma-4-E2B** — native tool-call template, and small enough that a six-arm sweep
fits the budget; ablations mattered more here than absolute score. **LoRA over full fine-tuning** —
arms differ only in an adapter, so merging gives directly comparable checkpoints. **GRPO over PPO** —
no value head, which at this scale is a meaningful fraction of the parameters. **A10** — the largest
tier reachable without a payment method, and the right call anyway (§6).

## 4. Honest evaluation: four things the headline hides

**(a) SFT made the model worse, and is still worth +0.301.** 0.132 → 0.035, for a diagnosable
reason: APIGen-MT teaches the model to ask a user simulator for missing information, and this
environment has none, so asking ends the episode having done nothing (50.4% of SFT episodes make no
tool call). Yet at matched compute RL reaches 0.496 from scratch and **0.797 from the SFT adapter**.
*Judged as a policy it should be deleted; judged as an initialisation it is decisive.* Only the
RL-only ablation separates those two claims — without it, 0.035 is just an embarrassing number.

**(b) The highest score in the table is a reward hack.** RL-only at 200 steps scores 0.929 by
skipping user identification in **98.4%** of write episodes and firing the write with the id leaked
in the prompt — exploiting a reward that scores final state but never the policy. Its lookup
compliance is 0.02 against base's 0.14. The compliance metric is what exposed it, and it *moves
opposite to success*, which is the general lesson: log the quantities the reward does **not** score.

**(c) A benchmark bug, and a circular test that hid it.** `return_items` read exactly 0.93 in three
independently trained arms — that coincidence was the tell, and 1.2% of tasks turned out to be
unwinnable. Worse, the test meant to prevent exactly this built its fixture *from the thing it
verified*, so it stayed green on 28 unsolvable tasks. **A test whose fixture derives from the code
under test proves nothing.** Both fixed and pinned; all six arms re-baselined.

**(d) None of it transfers.** Two external benchmarks, predictions committed beforehand
([BFCL](BFCL_PREREGISTRATION.md), [τ-bench](TAU2_PREREGISTRATION.md)).

| BFCL (2,340 cases) | Base | + SFT | SFT+RL (30) | RL only (200) |
|---|---|---|---|---|
| AST pooled | **0.807** | 0.371 | 0.493 | 0.795 |
| `parallel` | 0.725 | **0.000** | **0.000** | 0.715 |

**No trained arm beats base on anything, and parallel calling falls to exactly 0 of 200.** Syntax
stays valid; the model emits one call where two are required, because both the SFT corpus and the
environment are one-call-per-turn. A capability base *had* was trained out of it, and no in-domain
metric could see it — `tau-retail-lite` cannot even express the failing task. Notably **on-policy RL
preserved what imitation destroyed** (0.795 vs 0.807) despite training in the same environment. The
diagnosis was actionable: mixing in 500 parallel-call trajectories recovers `parallel` to 0.705 and
82% of the AST damage, confirming a missing demonstration rather than lost capacity.

On **τ-bench retail** — the benchmark `tau-retail-lite` is a simplification of — every arm solves
**0 of 35** tasks requiring a database write. All eight solves in the run are the three tasks whose
correct end state is an *unchanged* DB, and two were earned by firing a forbidden write six or seven
times and having the API reject each one: `db_check` compares a hash, so a blocked write and a
correct refusal score identically. That is finding (b)'s failure mode in a benchmark I did not write.
What survives is behavioural: **RL raises looping to death (0.394 → 0.850, *p* = 0.00007), lowers
asking, and eliminates escalation entirely** — the RL arms call `transfer_to_human_agents` zero times
where base manages 3/5, though `grpo_1500` scores 0.99 in-domain on a family requiring that exact
call.

## 5. One axis explains most of it

Order the arms by how well they abstain and the willingness-to-call column returns in near-perfect
reverse. `grpo_1500` is the extreme: best-in-class at calling when calling is right (BFCL
should-call **0.938**, above base's 0.625) and **less than a third as good at staying silent**
(0.225 vs 0.801). **The project's best agent is its worst abstainer.**

The cause is not what I first wrote down. The environment *does* pay full reward for refusing — but
only via `transfer_to_human`, a state-changing call, so a text-only decline scores 0.0 and successful
refusals average 4.5 tool calls. A fifth of RL training teaches that an out-of-scope request warrants
four to five calls. **That requirement exists because it patched an earlier hack where doing nothing
scored 1.0** — a reward-hack fix installed the bias an external benchmark found later. Adding an
abstention family recovers restraint (0.460 → 0.545, *z* = 4.06) for 0.039 of `pass^1`, so the axis
is separable — but the fix saturates in-domain at 400/400 while closing only 25% of the external gap.

## 6. Engineering findings

Eight are listed in [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md), and they sort by how they announce
themselves — **three that stay green while wrong** (Qwen3's template has no `{% generation %}`
marker, so botched masking silently trains the model to *generate* database contents), **four that
fail loudly but blame the wrong thing** (a Modal spend limit surfaces as "waiting for capacity"),
**and one that was never a bug at all.** None is caught by reading the logs: the first stays green,
the second lies, the third never produces a line.

That last one was a purchase about to be made on the wrong axis. **Concurrency, not a bigger GPU, was
the throughput lever**: 1.2 GB of weights against an A10's 600 GB/s is latency-bound at low
concurrency, never compute-bound. Raising BFCL's own thread cap to 64 gave 2.3× free and fanning the
arms across containers cut a sweep to 24 minutes at identical GPU-seconds, where an H100 costs
3.6×/hour to buy back the one thing that was not in the way.

## 7. Limitations

- **The environment leaks its own answers.** The `easy` instruction states the order and item ids, so
  the intended lookup is never *required* — only the policy text asks for it. The most serious flaw
  here, and what makes finding (b) possible.
- **One model, one environment, one seed per training run,** and the reward's shaping terms were
  never ablated. No claim here is known to survive a change to any of them.
- **τ-bench's `pass^1` is not a capability measurement** at 3 solves against 1, on tasks that reward
  inaction, with a user simulator that invented order ids in ~7.5% of conversations.

## 8. What a week would buy

1. **Fix the leak, then re-run everything.** Withhold ids from the instruction so lookup is
   *required*. It is the single change that would let 0.880 be described as tool use.
2. **Ablate the reward.** Every shaping term in §2 is a claim. Drop each in turn and watch both
   success and whether the term induces hacking — §4(b) shows construction and unit tests do not
   catch an exploit that only a long run surfaces.
3. **Train on the axis, not around it.** §5 shows act/abstain is separable, but the fix saturates
   in-domain while closing a quarter of the external gap; the abstention family needs BFCL-level
   diversity, not one hand-written topic list.
4. **Instrument for hacking by default, and audit fixtures for circularity.** The compliance metric
   that exposed §4(b) was written *after* the run that needed it — rising success with a falling call
   count should be an alarm, not progress.

## 9. How the work was directed

Written with an AI coding agent, which makes the interesting question what was *decided* rather than
what was typed. The round-by-round ledger is [FINDINGS.md](FINDINGS.md) §6; the loop closed in round
0, inside the cap, and every round after it was a choice about where to spend a $30 budget. Three of
those choices did most of the work.

**Buy external validation before polishing internal numbers.** BFCL sat in the plan unexecuted while
the in-domain table kept improving. Running it (round 3) is what converted "RL improved the model
6.7×" into "the improvement does not leave the environment it was trained in" — it cost this project
its headline and is the reason the document has a finding rather than a score.

**Never pay for a measurement without a free rehearsal.** The metered τ-bench run was preceded by a
zero-cost dry run pointing the user simulator at the local server. It measured nothing and caught two
harness bugs, one of which — τ-bench reporting `reward = 0.0` for conversations it never scored —
would have silently falsified every number in §4(d).

**Keep the full record; condense last.** The long version was written first and deliberately not
trimmed while results were still arriving, which is why refuted predictions (P17, P21, P23) survive
in the pre-registration files instead of being quietly reshaped into the ones that held. This
three-page document is the last step, not the first draft.
