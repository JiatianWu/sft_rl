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
- **User simulator:** `gpt-4.1-mini`, held fixed across arms. This is a deliberate cost compromise
  and it **breaks comparability with published leaderboard numbers**, which use frontier-class user
  models. Arm-versus-arm comparisons remain valid; any comparison to a published τ-bench score does
  not, and the writeup will say so wherever a number appears.
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
