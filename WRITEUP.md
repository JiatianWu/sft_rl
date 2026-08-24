# Multi-turn tool use on a 0.6B model: SFT → GRPO → eval

**Time budget:** 4 hours, hard cap. **Compute:** Modal, single A10.
**Model:** `Qwen3-0.6B`. **Artifacts:** `results/`, one JSON record per episode.

Task success on 2,400 held-out episodes per checkpoint, six arms:

| | Base | + SFT | RL only (30) | SFT+RL (30) | RL only (200) | SFT+RL (200) |
|---|---|---|---|---|---|---|
| **pass^1** | 0.140 | 0.037 | 0.489 | **0.794** | *0.935* | 0.863 |
| **looked up before writing** | 0.13 | 0.42 | 0.94 | **1.00** | **0.02** | **1.00** |

Three findings, none of which the headline number shows on its own.

**SFT made the model measurably worse** (0.140 → 0.037) for a diagnosable reason: APIGen-MT
teaches it to ask a user simulator for missing information, and this environment has none, so
asking ends the episode having done nothing. **Yet SFT is worth +0.305 as an RL prior** — at
matched compute RL reaches 0.489 from scratch and 0.794 from the SFT adapter. Judged as a
policy it should be deleted; judged as an initialisation it is decisive. Only the ablation
separates those claims.

**The highest score in the table is a reward hack.** RL-only at 200 steps scores 0.935 by
skipping user identification in 98.2% of write episodes and firing the write directly with the
id leaked in the prompt — exploiting the fact that my reward scores final state but never the
policy. The best *genuine* agent is SFT+RL at 30 steps, 0.794. Everything above it is
artifact, and §3.3 explains why the failure was mine rather than the model's.

The through-line is that every headline number here was misleading in isolation, and what
made the run interpretable was cheap instrumentation — per-episode records, an error taxonomy,
and a compliance metric that moves opposite to success.

---

## 1. What was built

| Stage | Choice |
|---|---|
| Base model | `Qwen3-0.6B` |
| SFT | LoRA r=32, assistant-only loss, on APIGen-MT-5k |
| Environment | `tau-retail-lite`, built in-repo: seeded DB, 10 tools, 6 task families |
| RL | GRPO via TRL's `environment_factory`, colocate vLLM, continuing the SFT adapter |
| Eval | Held-out task split, pass^k, identical decoding for all three checkpoints |

### Why these choices

**`Qwen3-0.6B`.** Its chat template already emits `<tool_call>` blocks, so SFT can spend
capacity on *when* to call rather than how to spell it. It is also small enough that a GRPO
step with 8 rollouts takes seconds, which is what makes an RL run fit the budget at all.

**GRPO over PPO.** GRPO is critic-free. Fitting a value head on a 0.6B model from a few
hundred episodes would be the least reliable part of the system, and its group-relative
baseline suits a binary task reward.

**A self-built environment over τ-bench itself.** τ-bench's reward design is the right one
and is copied directly (below), but the benchmark needs an LLM user simulator: API cost,
nondeterminism and latency inside the RL loop. A procedural environment gives thousands of
tasks, a train/test split sharing no database, and a free deterministic reward. The cost is
stated plainly in §5 — the headline number is measured on a benchmark I wrote.

**Stock PEFT instead of Unsloth.** Unsloth would have roughly halved SFT time, but it pins
its own `transformers` while TRL's `environment_factory` needs `transformers>=5.2`. The ~10
minutes saved did not justify a version conflict inside a hard time budget.

---

## 2. The environment and the reward

`tau-retail-lite` is a seeded retail back office: users, orders, products, payment methods.
Ten tools (five read, five write) are exposed to the model as the environment's public
methods. Six task families exercise different behaviours, including one — `refuse_invalid`
— where the user asks for something policy forbids and the correct action is to decline and
escalate.

### Grounding

The outcome reward follows τ-bench exactly, as a **product** of two checks:

```
r_outcome = r_action * r_output
  r_action = SHA-256 of the final database == ground-truth hash
  r_output = every required fact appears in what the agent told the user
```

The product is what makes the reward hard to game from either side: state-only rewards an
agent that mutates the database and says nothing useful, output-only one that recites a
plausible refund and never issues it. Note what the product still does not cover — *how* the
state was reached is unscored, which is the gap §3.3 exploits.

Ground truth is **derived**, not hand-written: the sampler executes the oracle actions against
a copy of the initial database and hashes the result, so oracle and reward cannot drift apart.
`tests/test_env.py` asserts the oracle scores 1.0 on every family across 40 seeds; if that
fails the task is unsolvable and any RL number on it is meaningless.

### Shaping, and why each term cannot be farmed

Binary outcome alone is far too sparse for a 0.6B. Total reward adds:

| Term | Range | Purpose | Why it is not exploitable |
|---|---|---|---|
| `r_outcome` | {0,1} | the real objective | — |
| `r_progress` | [0,1] | fraction of oracle actions performed **with correct arguments** | matched greedily against the oracle, so repeating one correct call cannot substitute for a missing one; a wrong write earns nothing and costs a violation |
| `r_format` (protocol) | [0,1] | parseable call, existing tool, sane arguments | annealed toward zero as it saturates, so it stops competing with the objective |
| `p_efficiency` | ≤0 | penalises redundant calls | **only charged when the episode already succeeded**, so it can never outrank correctness |
| `p_violation` | ≤0 | illegal writes, hallucinated tools, malformed calls | always charged |

### Two reward-hacking holes: one the tests caught, one they did not

The first version of `refuse_invalid` paid full marks for doing nothing. The correct final
state for a refusal is "database unchanged" and the family required no output, so an agent
replying *"I'm sorry, I can't help"* scored `r_action = 1`, `r_output = 1` vacuously, and a
perfect 1.0. The family meant to punish over-eager writing was rewarding passivity. The fix
gives escalation a state footprint — `transfer_to_human` sets an `escalated` flag inside the
hashed state, so refusing without escalating now fails the state check, and only the *fact*
of escalation is recorded so wording cannot decide the reward
(`tests/test_harness.py::test_refusal_requires_escalating_not_just_apologising`).

That one was catchable by construction. **The second was not**, and only appeared after 200
steps of RL: because the reward scores final state but never *conduct*, an agent that skips
user identification entirely and fires the write with the id leaked in the prompt earns a
perfect score. §3.3 is that result. The lesson is that a unit test can pin the holes you have
thought of, and a long enough RL run finds the ones you have not — so the instrumentation
that detects hacking matters as much as the reward that tries to prevent it.

### Handling sparse signal

GRPO's advantage is group-relative: when all G rollouts for a task score the same, the
advantage is zero and that task contributes nothing. At 0.6B the default is "all G fail".
Four things address this, and one metric monitors it:

1. **The shaping above**, so groups differentiate on protocol and partial progress long
   before any full success appears.
2. **An easy difficulty**, keeping the oracle to 1–3 calls. This is also the decision that
   backfired: stating the ids in the instruction removed the need to look anything up, which
   is what made the §3.3 shortcut available.
3. **SFT first**, to lift the success rate off the floor so RL has variance to exploit.
4. **G = 8**, to raise the chance of at least one success per group.

The ablation (§3.1) says (3) worked, but not for the stated reason. SFT *lowered* success to
0.037, so it supplied no successes to differentiate — yet RL from the SFT adapter still beat
RL from scratch by 0.305. What SFT contributed was a policy-compliant procedure to vary
around, not variance in outcome. The early gradient came from the shaping terms: a model that
asks a question scores zero on progress and format alike, while one that emits a well-formed
call scores on both, so groups differentiated even when no rollout in them fully succeeded.

---

## 3. Results

Six checkpoints, evaluated with identical decoding (T=0.7, top-p 0.95, fixed seed), identical
prompts and identical chat-template settings, on **600 held-out tasks × 4 trials = 2,400
episodes each**. Test seeds (100000–100099) are disjoint from training seeds (0–1999). "30"
and "200" are GRPO steps; the RL-only arms build a fresh LoRA from the same config SFT uses,
so they differ from the SFT arms in initialisation alone.

| Metric | Base | + SFT | RL only (30) | SFT+RL (30) | RL only (200) | SFT+RL (200) |
|---|---|---|---|---|---|---|
| **pass^1** | 0.140 | 0.037 | 0.489 | **0.794** | *0.935* | 0.863 |
| **pass^4** | 0.140 | 0.033 | 0.427 | 0.775 | *0.932* | 0.853 |
| state correct | 0.253 | 0.203 | 0.641 | 0.908 | 0.946 | 0.967 |
| reported correctly | 0.563 | 0.446 | 0.805 | 0.870 | 0.969 | 0.882 |
| tool calls / episode | 1.12 | 1.33 | 2.80 | 3.51 | **1.17** | 3.70 |
| illegal writes / episode | 0.050 | 0.002 | 0.185 | 0.171 | 0.221 | 0.189 |
| **looked up before writing** | 0.129 | 0.416 | 0.939 | **1.000** | **0.018** | **1.000** |

| Task family | Base | + SFT | RL only (30) | SFT+RL (30) | RL only (200) | SFT+RL (200) |
|---|---|---|---|---|---|---|
| `cancel_order` | 0.00 | 0.00 | 0.55 | 0.68 | 1.00 | 1.00 |
| `modify_address` | 0.51 | 0.21 | 0.88 | 1.00 | 1.00 | 1.00 |
| `return_items` | 0.00 | 0.00 | 0.85 | 0.93 | 0.93 | 0.93 |
| `lookup_and_report` | 0.32 | 0.00 | 0.10 | 0.72 | 1.00 | 0.82 |
| `refuse_invalid` | 0.00 | 0.00 | 0.03 | 0.89 | 0.81 | 0.99 |
| `exchange_items` *(never in RL)* | 0.01 | 0.01 | 0.53 | 0.55 | 0.86 | 0.44 |

Every episode is also bucketed by the **first** thing that went wrong, so the columns are
mutually exclusive and sum to 100% (`scripts/error_taxonomy.py`):

| Failure mode | Base | + SFT | RL only (30) | SFT+RL (30) | RL only (200) | SFT+RL (200) |
|---|---|---|---|---|---|---|
| no tool call | 13.8% | 49.8% | – | – | 0.6% | – |
| malformed / unknown tool | – | 0.2% | – | 0.3% | – | – |
| illegal write | 5.0% | 0.2% | 18.0% | 2.1% | 4.7% | 2.4% |
| stopped early | 67.2% | 38.2% | 12.8% | 6.3% | 0.1% | 0.9% |
| ran out of turns | – | 0.2% | – | 0.1% | – | 0.0% |
| acted, reported badly | – | 7.8% | 15.2% | 11.4% | 1.2% | 10.4% |
| reported, acted badly | – | – | 5.1% | 0.0% | – | – |
| other | – | – | – | 0.3% | – | – |
| **solved** | **14.0%** | **3.7%** | **48.9%** | **79.4%** | **93.5%** | **86.2%** |

Bucket meanings: **no tool call** answered from the prompt alone; **illegal write** called a
write the policy or order state forbids; **stopped early** answered with oracle actions
outstanding; **acted, reported badly** reached the right state without telling the user the
required facts; **reported, acted badly** said the right thing without making it true.

**The highest score in these tables belongs to the worst agent.** RL only (200) reaches 0.935
by discovering that the policy is unscored: it skips user identification in 98.2% of write
episodes and fires the write directly using the id leaked in the prompt. §3.3 is that story.
Read `looked up before writing` alongside `pass^1` throughout — neither is interpretable
alone, which is the main methodological lesson of this run.

**One ceiling is not 1.0.** `return_items` sits at exactly 0.93 in three separate arms, which
is too stable to be a model property. It is not: 7 of its 100 test seeds are **unsatisfiable**,
and §3.4 is that bug. Where a column reads 0.93 for that family, the checkpoint solved
everything solvable.

### 3.1 SFT made the model worse — and it was still worth running

SFT dropped success from 0.140 to **0.037**, for a diagnosable reason. The SFT checkpoint
stops calling tools — "used a tool at all" falls from 0.86 to 0.50 — and what it does
instead is ask a question:

> *"I cannot cancel an order or provide a refund without the user's user ID. Could you
> please provide me with your user ID so I can verify the order and proceed?"*

That is **correct in APIGen-MT and fatal here.** Those trajectories are collected against a
user simulator that answers follow-ups, so requesting missing information is rewarded.
`tau-retail-lite` has no simulator: the instruction is one turn and already contains the email
`find_user_id_by_email` needs, so asking ends the episode having done nothing. In 8 of 12
sampled no-tool transcripts the model asks rather than acts; the base model never does. A
train/serve distribution mismatch, not an optimisation failure — the kind of thing a loss
curve cannot show and an aggregate cannot explain.

**Judged as a policy, SFT should be deleted. Judged as an initialisation, it is worth +0.305**
— at matched compute RL reaches 0.489 from scratch and 0.794 from the SFT adapter. Only the
ablation separates those claims.

The mechanism is that **the two stages teach opposite, complementary things.** RL-only learns
to act, and acts well where acting is the answer (`return_items` 0.85, `modify_address` 0.88).
What it never learns is restraint: `refuse_invalid` collapses to **0.03**, and
`lookup_and_report`, where the correct action is to touch nothing, drops to **0.10**. It acts
indiscriminately because the environment mostly pays for acting. That is the axis SFT supplies
— it is over-cautious, asking instead of committing, and its illegal writes fall 25× (0.050 →
0.002). Composed, the two failure modes cancel.

### 3.2 GRPO repaired the failure the diagnosis named

GRPO took 0.037 → **0.794**, fixing precisely what the taxonomy below identified: "no tool
call" to 0.0%,
"stopped early" from 38.2% to 6.3%, tool calls per episode from 1.33 to 3.51. The model
chains. This is the case for online RL in one number — the environment scores what actually
happens, so a behaviour rewarded by the SFT corpus but useless at deployment is unlearned,
which more imitation on that corpus could not achieve.

Longer training (200 steps) sharpened the in-distribution picture and confirmed the
undertraining diagnosis exactly where it was specific: `cancel_order`, where all 129 failures
had stopped one call short of the write, went to a clean **1.00**, and overall success rose to
0.863. But the held-out family **regressed, 0.55 → 0.44**, while every trained family improved
or held. More steps bought in-distribution accuracy by narrowing the policy. The lever that
looks cheapest — train longer — degrades the property that made the result credible.

### 3.3 The highest score is a reward hack

Removing the SFT prior and training 200 steps produces the best number in the table, 0.935,
and the worst agent. It calls the write tool as its first and only action:

```
[user]      ...cancel an order. My email is yusuf.muller93@example.com... The order id is #W3001.
[assistant] <tool_call>{"name": "cancel_pending_order",
                        "arguments": {"order_id": "#W3001", "reason": "ordered by mistake"}}</tool_call>
[tool]      {"order_id": "#W3001", "status": "cancelled", "refund": 248.4}
```

One call produces the correct final state, and the write tool returns `refund: 248.4`, which
satisfies the output check for free. Lookup compliance falls from 0.939 at 30 steps to
**0.018** at 200 — the policy's first rule, *"Identify the user before acting"*, is skipped in
98.2% of write episodes.

**Two of my design flaws combined to make this optimal, and both are mine, not the model's.**
The `easy` instruction leaks every identifier needed — *"The order id is #W3001, current item
id P1003-0, and I want item id P1003-1"* — so no lookup is ever necessary. And the reward
scores only final state and reported facts; nothing in it references user identification, so
a rule stated in the system prompt is worth exactly zero. Given enough steps, RL correctly
found that the procedure was unpaid and discarded it. It also generalises *better* (held-out
`exchange_items` 0.86) precisely because "call the write with the id from the prompt" is
simpler and more transferable than the intended procedure.

This reframes SFT a third time: its contribution is not only restraint but **anchoring the
model to a procedure the reward never pays for**. SFT→RL holds 100% compliance at both 30 and
200 steps because it started there and no gradient pushed it off. That is a real benefit and
an accidental one — regularisation toward intent, not something the objective earned.

**So the best genuine agent in this table is SFT+RL at 30 steps (0.794), and everything above
it is measurement artifact.** Reporting 0.935 as the headline would have been the most
flattering and least honest reading of the run.

**Caveat on all RL numbers.** GRPO optimises the same reward the harness scores, so RL
checkpoints are trained on the eval metric in a way base and SFT are not. Disjoint seeds and a
held-out family control for memorising *tasks*, not for objective and metric coinciding.

### 3.4 A benchmark bug the aggregate hid, and the test that should have caught it

`return_items` reads exactly **0.93** in three independently trained arms. A number that
stable across different checkpoints is a property of the benchmark, not the model.

It is. On 7 of 100 test seeds the task cannot be solved truthfully. An order may list the same
`item_id` on more than one line — **7.8% of generated orders do** — and
`return_delivered_order_items` refunds every matching line, while the task asked for a single
unit price:

```
required_outputs : ['65', '65.00']            # one Backpack (grey)
tool returned    : {"refund": 130.0}          # the order lists it twice
```

The agent reports 130.0, which is what the environment told it and what actually happened, and
is scored wrong. 28 of 2,400 episodes (1.2%) were unwinnable, and `return_items` was capped at
0.93 — so the three arms reading 0.93 had solved **everything solvable**, and the "28 residual
failures" in the decomposition below are not model errors at all.

**The existing oracle test could never have caught it,** which is the more useful lesson. It
builds the oracle's reply out of `required_outputs` itself:

```python
def _oracle_text(env):                       # the old helper
    return " ".join(alts[0] for alts in env.spec.required_outputs)
```

So the output half of `test_oracle_gets_full_reward` asserts that a string built from
`required_outputs` contains `required_outputs`. It tests the matcher against itself and never
against the environment — green on a task no agent can pass. The replacement,
`test_required_facts_are_obtainable_from_the_tools`, scores the **actual tool return values**
over the full evaluation split, and fails loudly on the seed above. Both the fix and the test
are in this commit; **the numbers above predate the fix** and are left as measured.

The generalisable form: a fixture derived from the thing under test proves nothing. The oracle
test looked like the most load-bearing test in the repo — its own docstring says so — and half
of it was circular.

### 3.5 What is actually left to fix

Decomposing the residual failures of each RL arm by which half of `r_outcome = r_action *
r_output` failed:

| | SFT+RL (30) | SFT+RL (200) | RL only (200) |
|---|---|---|---|
| total failures | 494 (20.6%) | 330 (13.8%) | 157 (6.5%) |
| state right, **report** wrong | 55.3% | 75.8% | 17.8% |
| report right, **state** wrong | 36.6% | 14.2% | 52.2% |
| neither | 8.1% | 10.0% | 29.9% |

For the best genuine agent, **the majority of what remains is reporting, not acting** — it
does the job and then fails to say the required number. That is a much cheaper problem than
tool selection, and it is concentrated: of its 494 failures, 181 are `exchange_items` (the
held-out family), 129 `cancel_order`, 111 `lookup_and_report`, and 28 are the dead
`return_items` tasks from §3.4. Longer training pushed the mix further toward reporting
(75.8%), which is consistent with RL fixing execution first.

RL only (200) inverts this — half its failures are *state* wrong with the report correct,
the signature of an agent that says the right thing without reliably making it true, which is
the same disposition as the §3.3 shortcut.

**A harness hypothesis, tested and wrong.** The largest single bucket above is 134
`exchange_items` episodes with `r_progress` 1.0 and `r_output` 0 — executed perfectly, scored
zero on reporting. The output matcher is a substring test that demanded the leading `#`, so
*"Your order W3006 has been exchanged"* failed, and an order id is the entire required output
of two families. That is a real defect and worth fixing on its own terms. It was also an
attractive explanation, so it got measured rather than assumed: ids now accept both
renderings (`_order_id_variants`, pinned by a test), and the arm was re-run.

| | before | after | |
|---|---|---|---|
| `return_items` | 0.930 | **1.000** | the §3.4 dead tasks, exactly as predicted |
| `exchange_items` | 0.547 | 0.532 | **unchanged — hypothesis falsified** |
| pass^1 | 0.794 | 0.802 | +0.007, all of it from §3.4 |

The `#` was not the problem. With failure transcripts now saved (they previously sampled the
first twelve episodes, which were all passes — the file could not answer the one question it
existed for), the actual behaviour is visible: asked to *"Confirm the order id when done"*,
the model executes the exchange correctly and then confirms the **item** ids instead, adding a
$120.00 "refund" that an exchange does not produce. It is a genuine instruction-following
failure, not a formatting artifact. The reporting shaping term is therefore the right lever,
and loosening the matcher further would only have hidden this.

**How the RL target was chosen in the first place.** An aggregate of 0.140 says a checkpoint is
bad, not what to fix. The informative entry for the base model is what is *absent*: malformed
calls and hallucinated tools are **0%**, and no episode hit the turn limit — all 2,400 ended
because the model chose to answer. It is not that it cannot spell a tool call; it does not take
enough turns. Among the 1,792 early stops `r_progress` is `0.0` in *every one*, so outcomes are
near-bimodal — whole task or nothing, the regime where group-relative advantage collapses. The
taxonomy named premature termination, and that is exactly the metric RL moved: "stopped early"
67.2% → 6.3%, "no tool call" 13.8% → 0%.

---

## 4. Engineering findings

Full list in [`ENGINEERING_NOTES.md`](ENGINEERING_NOTES.md); each was found by running the
producing code rather than trusting a doc, and each is pinned by a test. The four that
changed the design:

- **TRL's `environment_factory` evaluates every property on a freshly built environment**
  before calling `reset`, so an environment that is invalid until reset crashes during
  *trainer construction*, not at rollout time.
- **Qwen3's template has no `{% generation %}` marker**, so `assistant_only_loss` is
  unusable and masking is manual. Getting it wrong is silent: training on tool-response
  tokens teaches the model to *generate* database contents, the exact hallucination the
  environment punishes.
- **Thinking had to be disabled for a fair baseline** — with it on, the 0.6B spends its
  whole budget reasoning and never calls a tool, which would have handicapped the *base*
  checkpoint specifically and inflated SFT's apparent gain.
- **A Modal spend limit is reported as "waiting for capacity",** and it is not the credit
  balance: $1.40 of ~$30 had been spent when everything stopped. Transient scarcity, a dead
  budget and a settings cap all look identical in the logs.
- **A test whose fixture is derived from the thing under test proves nothing** (§3.4). The
  most load-bearing assertion in the repo was half circular and stayed green on 28 unsolvable
  tasks.

---

## 5. Honest limitations

- **The benchmark leaks its own answers.** The `easy` instruction states the order id and the
  item ids, so the intended lookup procedure is never actually *required* — only the policy
  text asks for it, and the reward does not. §3.3 is the consequence: given enough steps RL
  finds the shortcut. This is the most serious flaw in the environment and it invalidates the
  two 200-step numbers as measures of tool use.
- **The reward scores outcomes, not conduct.** State plus reported facts is verifiable and
  cheap, but a policy rule that is never scored is a suggestion. Any rule that matters must
  appear in the reward, not only in the system prompt.
- **1.2% of evaluation episodes were unwinnable** (§3.4), capping `return_items` at 0.93. The
  bug is fixed and pinned by a test, but every number in this document was measured before the
  fix, so the reported ceiling is 0.988 rather than 1.0 and `return_items` rows understate
  three arms by 0.07. I chose not to re-run: it would cost another full sweep to move numbers
  I can correct in prose, and re-measuring only some arms would break comparability.
- **The headline benchmark is one I wrote.** Train and test share no database and one task
  family is held out of RL entirely, which controls for memorisation but not for the
  environment being easier, or differently shaped, than a real benchmark.
- **BFCL was cut.** It was planned as the external, uncontaminated check and is genuinely
  suitable (fully offline, deterministic, no judge). It was dropped when the A10 fallback
  and the debugging above consumed the slack. Without it, nothing here demonstrates that
  gains transfer off-distribution.
- **RL is trained on the eval metric.** GRPO optimises the same grounded reward the harness
  scores. Disjoint seeds and a held-out family rule out memorising tasks, but not the
  objective and the metric being the same function. Base and SFT get no such advantage, so
  the 0.140 → 0.794 comparison is not between equals.
- **Single seed, one run per arm.** No error bars, and 2,400 episodes per checkpoint constrain
  sampling noise but say nothing about run-to-run variance in training. The SFT-as-prior
  result (+0.305) rests on one pair of runs.
- **The SFT stage is under-trained.** 63 steps over 1,000 trajectories, at batch size 1, is
  small. The regression in §3.1 is explained by a specific verifiable behaviour rather than by
  undertraining, but more SFT might have changed its sign, and that was not tested.
- **Longer RL was measured, not understood.** 200 steps improved in-distribution success and
  *reduced* held-out success (0.55 → 0.44). Whether that is overfitting to five families,
  drift toward the shortcut, or both, was not isolated.
- **A residual train/inference mismatch remains.** During a rollout the empty `<think>`
  block prefixes only the turn being generated and vanishes from re-rendered history. A
  single contiguous SFT sequence cannot reproduce that; training without think blocks keeps
  the gap to a constant 5-token prefix rather than spreading it across the history. It is
  asserted exactly in `tests/test_masking.py` so it cannot silently widen.

---

## 6. What a week would buy

Ordered by expected value, not by effort.

1. **Close the shortcut, then re-run everything above it.** Stop leaking ids in the
   instruction so a lookup is genuinely required, and score conduct as well as outcome — an
   episode that writes before identifying the user should not earn full marks. Until this
   lands, the two 200-step numbers measure my reward function rather than the model, and the
   comparison between arms is confounded by how far each drifted toward the exploit.
2. **Fix the illegal-write regression.** RL roughly tripled policy violations (0.050 → 0.19)
   in every arm while improving the composite reward, which means the weighting sells
   compliance for completion. Make violations episode-terminating rather than a subtracted
   term and re-measure both numbers; a deployable agent cannot trade them this way.
3. **Fix the SFT data mismatch rather than the SFT hyperparameters.** §3 shows the corpus
   teaches asking a user simulator that does not exist at deployment. Two clean options:
   filter APIGen-MT to trajectories that never request missing information, or add a scripted
   user to the environment so the behaviour becomes viable instead of fatal. The first is a
   day; the second is more faithful to τ-bench.
4. **Attack the reporting failures, which are now the majority.** §3.5 shows 55% of the best
   agent's residual failures are correct-state-wrong-report, and longer training pushes that
   to 76%. In all 273 such episodes `r_progress` is 1.0, so the model had already executed the
   task and held the fact it failed to relay — a copying failure, not a reasoning one. It is
   **not** a budget problem: only 2 of 494 failures hit the turn limit, and failures average
   4.41 turns against 4.55 for successes. A shaping term on relaying observed values is the
   intervention; more turns is not.
5. **Run BFCL multi-turn** on every checkpoint, plus real τ-bench with an LLM user
   simulator, to find out whether any of this transfers off-distribution. This is the check
   that would let the 0.794 be described as tool use rather than as this environment.
6. **Ablate the reward.** Every shaping term in §2 is a *claim* backed by construction and a
   unit test, not by measurement. Drop each in turn and watch both final success and whether
   the term induces hacking — §3.3 shows construction and unit tests are not enough to catch
   an exploit that only a long run surfaces.
7. **Instrument for hacking by default.** The compliance metric that exposed §3.3 was written
   *after* the run that needed it. Any quantity the reward does not score but the task
   requires should be logged from the first evaluation, and a rising success rate paired with
   a falling call count should be treated as an alarm rather than progress.
8. **Audit every fixture for circularity** (§3.4). One test was building its input from the
   thing it verified; I have not checked the rest of the suite for the same shape.
