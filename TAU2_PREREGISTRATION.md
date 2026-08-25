# τ-bench pre-registration

Written and committed **before** running τ-bench, on the same terms as `BFCL_PREREGISTRATION.md`:
every claim is stated with the observation that would refute it, and all outcomes get reported in
`WRITEUP.md` whether they support the existing conclusions or not.

Prediction numbering continues from the BFCL file (P1–P17) so the two can be cross-referenced from
`WRITEUP.md` without collision.

## What this measures, and what it does not

**τ-bench retail is not an independent benchmark for this project, and the writeup must not present
it as one.** `tau-retail-lite` was built as a simplification of τ-bench retail, and the overlap is
closer than "same domain": seven of this environment's tool names — `find_user_id_by_email`,
`get_user_details`, `get_order_details`, `cancel_pending_order`, `modify_pending_order_address`,
`return_delivered_order_items`, `exchange_delivered_order_items` — appear **verbatim** in τ-bench
retail's 13-tool schema. The models were trained against nearly the real thing.

That is not contamination in the leakage sense; the environment was written from the paper's
description, not from its tasks. But it does change the question being asked. BFCL asked "did
general tool-use capability transfer?" and answered no. τ-bench retail asks something narrower and,
for this project, more pointed: **was the simplification faithful, and do the gains survive real
complexity?**

## Why run it anyway

**1. It can reverse the project's headline SFT result.** §3.1's central finding is that SFT *hurt*,
dropping pass^1 from `base`'s 0.132 to 0.035. The diagnosis was specific: APIGen-MT teaches the
model to ask a user simulator for missing information, and `tau-retail-lite` has no user simulator,
so roughly half of SFT's episodes make no tool call at all. **τ-bench has a user simulator.** If
that diagnosis is right, the SFT arms should do relatively better here — and if they beat `base`,
then "SFT hurts" is a fact about my environment rather than about SFT. No other benchmark available
to this project can test that, BFCL included.

**2. Its reward is structurally identical to the one in this repo.** τ-bench scores
`DB × COMMUNICATE` — a database-end-state check times a required-output check, multiplied.
`compute_reward` computes `r_action * r_output`, the same product, arrived at independently. τ-bench
also scores end-state only, deliberately not requiring the reference trajectory, matching this
repo's choice to hash the DB rather than match call sequences.

**3. It scores the abstention case that this environment cannot express.** τ-bench's docs state
plainly that an agent which does nothing but politely refuse receives full reward on tasks whose
correct answer is refusal — the DB is untouched, so its hash matches. §3.9 documented that I
patched exactly this case, giving `transfer_to_human` a state footprint to close a passivity hack,
and that the patch is *why* the RL arms never learned to abstain. τ-bench leaves the case open, so
it can measure the behaviour my own environment made unscoreable.

## What will be run

- **Domain:** `retail`, 114 tasks in the `base` split. Not airline or telecom — retail is the only
  domain with any relationship to what was trained.
- **Staged.** A 10-task pilot on `base`, `sft`, and `grpo_1500` first. The full 114 × 4 trials is
  bought only if the pilot returns a non-zero signal.
- **User simulator:** `Qwen/Qwen3.6-27B-FP8`, served on a Modal Endpoint and held fixed across
  arms. An open-weight customer **breaks comparability with published leaderboard numbers** more
  thoroughly than a smaller commercial model would. Arm-versus-arm comparisons remain valid; any
  comparison to a published τ-bench score does not, and the writeup will say so wherever a number
  appears.

  **This choice is a confound in one specific direction, and it is why a small model was not
  used.** A customer that cannot hold up its half of the conversation fails to volunteer the
  information the agent needs. That penalises every arm, but it penalises *asking* far more than
  *acting* — and asking is precisely the behaviour SFT learned and that P18 predicts will finally
  be rewarded here. A weak user simulator could therefore manufacture a false refutation of P18:
  not "§3.1's diagnosis was wrong" but "the customer was incapable of answering". The zero-cost
  plumbing run demonstrated the extreme case, where a 0.6B customer answered "I'm here to assist
  you with your request" and played agent instead. **The rate of normal conversation termination
  will be reported alongside every result as a health check on the customer**; if it is low, the
  comparison is not trustworthy in either direction.
- **Isolation:** a fourth Modal image. `tau2-bench` installs into its own uv-managed venv, so unlike
  the BFCL integration its dependencies never meet vLLM's.

**The one trap, cleared in advance.** Qwen3 emits `<tool_call>{...}</tool_call>` in the message
body. Without `--tool-call-parser hermes`, vLLM returns that as plain `content`, τ-bench records an
agent that never called a tool, and every task scores 0 — indistinguishable from "0.6B is too small
for this benchmark". This is the BFCL merge bug's exact shape: a clean, quotable, entirely false
result. `tau2_probe` checks it for free before any metered call, and has passed for `grpo_1500`
(`content: None`, one correctly-parsed `find_user_id_by_email` with the right argument).

## Predictions

**P18 — the load-bearing one. `sft` closes most of its in-domain deficit against `base`, and may
overtake it.** In-domain `sft` trails `base` by 0.097 (0.035 vs 0.132). §3.1 attributes that
entirely to the missing user simulator. With one present, I expect the gap to close to within 0.03
or invert. This is the prediction the whole run exists to test.

*Refuted if* `sft` trails `base` by a similar or larger margin than in-domain. That would mean the
§3.1 diagnosis was wrong — SFT damaged something real rather than being penalised by a missing
interlocutor — and §3.1 needs rewriting, not defending.

**P19 — absolute scores will be very low; pass^1 under 0.15 for every arm.** Frontier models score
around 0.6 on τ-bench retail. This is a 0.6B model on a multi-turn task requiring a correct DB end
state *and* specific communicated strings, as a product. I expect single digits.

*Refuted if* any arm clears 0.15, which would make this a far stronger result than the project
has any right to expect.

**P20 — `grpo_1500` will not lead by the margin it leads in-domain, and may not lead at all.**
It scores 0.880 in-domain against `base`'s 0.132, a 6.7x gap. But it was optimised against an
environment with no user simulator, where information is handed over rather than extracted. τ-bench
requires interrogating a user. RL never rewarded that, so I expect the in-domain advantage to
compress sharply, consistent with BFCL's verdict that these gains do not travel.

*Refuted if* `grpo_1500` reproduces anything like its in-domain dominance, which would be the first
evidence in this project of an RL gain transferring off-distribution.

**P21 — `grpo_1500` will lose reward specifically on refusal tasks.** It is the project's worst
abstainer (BFCL should-not-call 0.225). τ-bench retail contains tasks whose correct answer is to
refuse and leave the DB untouched. An agent trained never to abstain will write to the DB, change
the hash, and score 0 on precisely those tasks. This should be visible as a per-task pattern, not
just an aggregate.

*Refuted if* its failures are spread evenly across task types, which would mean the abstention
deficit measured by BFCL is a format artifact rather than a behavioural one.

## Observed during the zero-cost plumbing run, before any measurement

The chain was validated by pointing the *user simulator* at the same local vLLM server as the
agent, which costs nothing and exercises the real `user_simulator` code path. It is not a
measurement: a 0.6B customer answers "I'm here to assist you with your request", playing agent
rather than customer, so no conversation converges. Three things came out of it that matter here.

**Two harness bugs, found before they could cost anything.** Conversations overflowed a 16k context
window, which τ-bench records as an infra error and drops from the denominator. And more seriously,
τ-bench leaves `reward_basis`, `db_check` and `communicate_checks` **null** when a run ends on
`max_steps` or an error, while `reward` still defaults to `0.0` — so averaging naively over all
simulations manufactures a clean `pass^1 = 0.000` out of runs that were never scored at all. The
summary now separates scored from unscored and reports `null` rather than a fake zero.

**Agent-side plumbing is confirmed healthy.** Correct tool names, well-formed arguments, correct
schema, parsed cleanly — the failures are entirely downstream of having no real interlocutor.

**A hypothesis for P20/P21, logged now so it is not retrofitted later.** Given no usable
information, `grpo_1500` invented plausible placeholders (`find_user_id_by_email` called with a
*user_id*, then a fabricated name and zip) and then **repeated the same two failing calls ten times
until the error limit fired, rather than asking the user for the missing detail**. If that survives
contact with a competent user simulator, it is §3.9's finding showing up externally: an environment
that rewards acting and punishes deferring produces an agent that retries instead of asking. Stated
as a hypothesis, from one conversation with a broken counterpart — not as a result.

## Follow-up 5: P22, and why it is a new prediction rather than a rescue of P18

**The pilot's headline confirmed P19 and killed P18's testability.** Ten retail tasks against
`base`, `sft` and `grpo_1500` produced `pass^1 = 0.000` in every scored simulation of every arm,
with `DB = 0.000` throughout. Nobody solved anything. The pre-committed rule — "if the pilot returns
zero for all three arms, the full run is not purchased" — fires. P18 asked whether `sft` closes its
in-domain deficit against `base`; the deficit did close, from −0.097 to exactly 0.000, but only
because both ends collapsed to the floor. That is a degenerate confirmation and is recorded as
**not testable at this resolution**, not as a pass.

**What did discriminate was the failure mode, and it is not the metric this file pre-registered.**

| arm | loops to death (`too_many_errors`) | conversation ends normally |
|---|---|---|
| `base` | 7/10 | 2/10 |
| `sft` | **2/10** | **5/10** |
| `grpo_1500` | **10/10** | **0/10** |

Fisher exact, two-sided: `sft` vs `grpo_1500` on loop rate is 2/10 against 10/10, *p* = 0.0007.

Normal-stop rate was pre-registered above as a **health check on the customer**, not as an arm
metric. Promoting it to a headline result now, because it happens to point the way §3.1 predicts,
is precisely the post-hoc move this file exists to prevent. So the pilot finding is logged as
**exploratory**, and the claim gets a fresh prediction tested on data it has never seen.

**Design.** Tasks `10`–`49`, disjoint from the pilot's `0`–`9`. Four arms: `base`, `sft`, `grpo`
(SFT+RL 30) and `grpo_1500`. Simulations ending in `infrastructure_error` are excluded from all
denominators — declared here because the pilot's infra rate was uneven across arms (1, 3, 0) and
choosing that rule after seeing the split would be another way to cook the result.

**P22 — `sft` loops to death materially less often than `grpo_1500`, on fresh tasks.** The pilot
gap is 2/10 against 10/10; I expect it to narrow on 40 tasks but survive, with `sft` below 0.45 and
`grpo_1500` above 0.75.

*Refuted if* the rates come within 0.15 of each other, which would mean the pilot split was small-
sample noise and the exploratory finding should be dropped rather than written up.

**P23 — the mechanism is asking, and it will be visible directly.** The claim is not "SFT loops
less" but "SFT asks the customer for missing information instead of inventing it". So the rate of
agent turns that put a question to the customer, rather than calling a tool, is measured directly:
I predict `sft` asks at more than twice `grpo_1500`'s rate, and that asking rate tracks loop rate
across all four arms.

*Refuted if* loop rates differ while asking rates do not. That would mean the loops are generic
small-model repetition and the tidy §3.9 story — an environment that rewards acting and punishes
deferring produces an agent that retries instead of asking — is wrong, however well it reads.

**P24 — `grpo` sits between `sft` and `grpo_1500`, nearer the latter.** If RL against a
user-less environment is what erodes the asking behaviour, then applying it to the SFT checkpoint
should move that checkpoint toward the failure. This is the prediction that actually separates
"RL broke it" from "SFT fixed something base never had".

*Refuted if* `grpo` matches `sft`, which would mean 30 steps of RL leave the disposition intact and
`grpo_1500`'s collapse comes from its corpus rather than from RL.

**The alternative hypothesis, stated before the data.** `base` loops at 7/10 — high, untrained.
That is consistent with looping being the *default* behaviour of a 0.6B, with SFT fixing it and RL
undoing the fix. It is equally consistent with looping being generic repetition that correlates
with nothing. P23 is what tells these apart, and it is the reason the mechanism is measured
directly rather than inferred from termination reasons.

## Follow-up 5 outcome: P22–P24 on tasks 10–49

| prediction | verdict |
|---|---|
| **P22** `sft` loops < 0.45, `grpo_1500` > 0.75 | **confirmed** — 0.394 and 0.850, *p* = 0.00007 |
| **P23** `sft` asks > 2× `grpo_1500`, asking tracks looping | **refuted** — 1.42×, and `base` breaks the ordering |
| **P24** `grpo` between `sft` and `grpo_1500`, nearer the latter | **confirmed** — 0.718, and `sft`→`grpo` is *p* = 0.008 |

**The pilot's most quotable number was noise, and fresh tasks killed it.** The pilot had `sft`
looping 2/10 against `base`'s 7/10, which read as SFT fixing the behaviour. On 40 tasks the two are
0.394 and 0.424 — *p* = **1.00000**. SFT does not reduce looping at all. Running this on data the
claim had never seen is the only reason that is known.

**What survives is a cleaner claim than the one I set out to test.** Along the SFT→RL lineage both
quantities move monotonically, and in opposite directions:

| arm | loops to death | asks the customer | in-domain `pass^1` |
|---|---|---|---|
| `base` | 0.424 | 0.213 | 0.132 |
| `sft` | 0.394 | **0.422** | 0.035 |
| `grpo` | 0.718 | 0.348 | 0.797 |
| `grpo_1500` | **0.850** | 0.297 | **0.880** |

SFT roughly doubles asking over `base` (0.213 → 0.422) without touching the loop rate. RL then
walks asking back down and the loop rate up, step by step, and **the ordering is exactly the
in-domain ranking reversed**. Thirty GRPO steps applied to the SFT checkpoint nearly double its
loop rate (0.394 → 0.718, *p* = 0.008). That is P24, and it is the result that separates "RL broke
it" from "SFT fixed something `base` lacked": SFT fixed nothing here, RL broke it.

**P23's refutation is the useful part.** `base` asks least of all four arms (0.213) yet loops less
than either RL arm, so asking cannot be the whole mechanism. The tidy §3.9 story — an environment
that rewards acting and punishes deferring produces an agent that retries instead of asking —
holds *within* the SFT→RL lineage and fails across it. Asking is one lever, not the lever, and the
pre-registered refutation condition is what forced that qualification rather than letting the neat
version stand.

**The headline, on the honest denominator.**

| arm | solved / usable | in-domain `pass^1` |
|---|---|---|
| `base` | 3/33 = **0.091** | 0.132 |
| `sft` | 3/33 = **0.091** | 0.035 |
| `grpo` | 1/39 = 0.026 | 0.797 |
| `grpo_1500` | 1/40 = 0.025 | 0.880 |

**The in-domain ranking is inverted.** `grpo_1500` is this project's best agent in its own
environment by a factor of 6.7 over `base`, and on τ-bench retail it is the worst arm measured,
roughly 3.6× behind the untrained model.

**P18 is confirmed, on data that can finally test it.** In-domain `sft` trails `base` by 0.097
(0.035 vs 0.132); here they are identical at 0.091. Give the model the user simulator whose absence
§3.1 blamed, and SFT's deficit vanishes entirely. The caveat is 3 solved out of 33 either way, so
this is consistent with P18 and underpowered to distinguish "closed" from "small and unmeasured".

**P19 confirmed** — every arm below 0.15. **P20 confirmed and then some**: `grpo_1500` does not
merely fail to lead, it trails. **P21 adjudicated below** (`scripts/tau2_p21.py`).

## Follow-up 6 outcome: P21, and the thing it turned up by accident

Classifying the 40 tasks by their gold action list — refusal iff the gold terminates in
`transfer_to_human_agents`, τ-bench's analogue of the in-domain `refuse_invalid` family — gives 3
refusal, 2 read-only, 35 write.

**P21 is refuted, in the exact opposite direction to the prediction.** I predicted `grpo_1500` would
write to the DB on refusal tasks and score 0 on precisely those. It writes on no-write tasks *least*
of the four arms (1/5, against `base`'s 4/5), and refusal tasks are the **only** tasks it solves.

| arm | refusal (3) | write (35) | read-only (2) | escalated correctly |
|---|---|---|---|---|
| `base` | 2/3 | **0/35** | 1/2 | 3/5 |
| `sft` | 2/3 | **0/35** | 1/2 | 2/5 |
| `grpo` | 0/3 | **0/35** | 1/2 | **0/5** |
| `grpo_1500` | 1/3 | **0/35** | 0/2 | **0/5** |

**The half of P21 that survives is the escalation column.** Both RL arms call
`transfer_to_human_agents` **zero times** across all five no-write tasks, where `base` manages 3/5.
In-domain, `grpo_1500` scores **0.99** on `refuse_invalid`, a family that *requires* exactly that
call. So the abstention behaviour §3.10 trained is intact in-domain and does not fire once off it —
a sharper transfer failure than the aggregate showed, and the per-task pattern P21 asked for, just
not in the variable P21 named.

**Every arm solves 0 of 35 write tasks. All 8 solves in the whole run are tasks 10, 12 and 25** —
the ones whose correct end state is a DB that never changed. Nothing here demonstrates a 0.6B model
completing a retail transaction; the run measures how reliably an arm avoids doing damage.

**Two of `base`'s three solves were earned by being blocked.** On task 12 it attempted
`cancel_pending_order` six times and the API rejected all six; on task 10, seven rejected
`return_delivered_order_items` calls. `db_check` compares a hash, so a rejected write and a correct
refusal are indistinguishable to it, and both score 1.0. That is **§3.3's exact failure mode — a
reward that scores final state and never the policy — in a benchmark I did not write**, which is
some comfort about §3.3 and none at all about the 0.091.

**The user simulator is a live confound, at about 7.5%.** It invented order ids in 3/40, 3/40, 4/40
and 0/40 conversations per arm (`#900028208827` where the DB holds `#W5490111`), and drifts into
agent persona — on task 12 the *customer* says "Let me check the status of your order… I can cancel
it". Task 12 is one of the solves above: `base`'s writes failed with "Order not found" because the
customer supplied a fictional order. So the headline gap rests on 3 solves against 1, two of which
trace to a hallucinating counterpart.

**What this does and does not cost §3.12.** The `pass^1` column is weaker than it looked and should
not be read as a capability ranking. The loop-rate result is untouched: it is measured on all 40
tasks, needs no reward, and carries *p* = 0.00007 and *p* = 0.008. The behavioural findings — RL
raises looping, lowers asking, and abandons escalation entirely — are what this run actually
established.

**A denominator that would have reversed the headline.** τ-bench scores only conversations that
terminate normally, so averaging over *scored* simulations conditions on surviving — and surviving
is precisely what differs between these arms. That inflates `grpo_1500` from 1/40 to 1/6 = 0.167,
level with `sft`, making the worst arm look tied for best. A conversation that looped until the
error limit did not solve its task. `scripts/tau2_table.py` now reports both and leads with
solved/usable; the biased column is kept visible because it is the number a naive reading produces.

## What would make this uninformative

**All arms scoring 0/114.** P19 already predicts very low scores, and there is a real chance they
are *uniformly* zero, in which case the run discriminates nothing — the BFCL multi-turn power
problem (0.020–0.080 across four arms, differences well inside the 0.012–0.017 noise floor) with a
metered API attached. That is what the 10-task pilot is for, and the pre-committed decision is: **if
the pilot returns zero for all three arms, the full run is not purchased**, and the reported result
is "below the measurement floor of τ-bench retail for a 0.6B model" — which is an honest finding,
not a failed experiment.

**The subtler risk: P18 confirmed for the wrong reason.** If `sft` beats `base` here, the tempting
story is "the environment was the problem all along". But `sft` could also gain simply by talking
more, since `COMMUNICATE` rewards saying specific strings and SFT produces more conversational
output. Before claiming the §3.1 reversal, the DB and COMMUNICATE components must be reported
*separately*: a rise driven entirely by COMMUNICATE with a flat DB component means SFT got chattier,
not more capable. Stating this before seeing the numbers is the point of writing it here.
