# Multi-turn tool use on a 0.6B model: SFT → GRPO → eval

**Time budget:** 4 hours, hard cap. **Compute:** Modal, single A10.
**Model:** `Qwen3-0.6B`. **Artifacts:** `results/`, one JSON record per episode.

Task success on 2,400 held-out episodes per checkpoint, six arms, all measured on one metric:

| | Base | + SFT | RL only (30) | SFT+RL (30) | RL only (200) | SFT+RL (200) |
|---|---|---|---|---|---|---|
| **pass^1** | 0.132 | 0.035 | 0.496 | **0.797** | *0.929* | 0.868 |
| **looked up before writing** | 0.14 | 0.42 | 0.94 | **1.00** | **0.02** | **1.00** |

Three findings, none of which the headline number shows on its own.

**SFT made the model measurably worse** (0.132 → 0.035) for a diagnosable reason: APIGen-MT
teaches it to ask a user simulator for missing information, and this environment has none, so
asking ends the episode having done nothing. **Yet SFT is worth +0.301 as an RL prior** — at
matched compute RL reaches 0.496 from scratch and 0.797 from the SFT adapter. Judged as a
policy it should be deleted; judged as an initialisation it is decisive. Only the ablation
separates those claims.

**The highest score in the table is a reward hack.** RL-only at 200 steps scores 0.929 by
skipping user identification in 98.4% of write episodes and firing the write directly with the
id leaked in the prompt — exploiting the fact that my reward scores final state but never the
policy. The best *genuine* agent is SFT+RL at 30 steps, 0.797. Everything above it is
artifact, and §3.3 explains why the failure was mine rather than the model's.

**And on an external benchmark, none of it transfers.** BFCL over 2,340 test cases (§3.6, with
predictions committed beforehand): no trained arm beats base on any category, and SFT is
*catastrophic* — pooled AST accuracy 0.807 → 0.371, with parallel function calling collapsing to
exactly **0 of 200**. The model learned to emit one tool call and stop, because both my SFT
corpus and my environment are one-call-per-turn. RL-only, by contrast, preserves base capability
(0.795), so on-policy RL turns out to be far more conservative than imitation. The 0.797 is a
real gain in the environment it was trained for, bought with general capability that no
in-domain metric could see.

The through-line is that every headline number here was misleading in isolation, and what
made the run interpretable was cheap instrumentation — per-episode records, an error taxonomy,
a compliance metric that moves opposite to success, and finally a benchmark I did not write.

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
0.035, so it supplied no successes to differentiate — yet RL from the SFT adapter still beat
RL from scratch by 0.301. What SFT contributed was a policy-compliant procedure to vary
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
| **pass^1** | 0.132 | 0.035 | 0.496 | **0.797** | *0.929* | 0.868 |
| **pass^4** | 0.132 | 0.033 | 0.443 | 0.782 | *0.922* | 0.863 |
| state correct | 0.245 | 0.200 | 0.638 | 0.899 | 0.929 | 0.963 |
| reported correctly | 0.563 | 0.448 | 0.812 | 0.879 | 0.973 | 0.888 |
| tool calls / episode | 1.13 | 1.32 | 2.78 | 3.52 | **1.17** | 3.71 |
| illegal writes / episode | 0.047 | 0.000 | 0.186 | 0.169 | 0.240 | 0.193 |
| **looked up before writing** | 0.137 | 0.415 | 0.936 | **1.000** | **0.016** | **1.000** |

| Task family | Base | + SFT | RL only (30) | SFT+RL (30) | RL only (200) | SFT+RL (200) |
|---|---|---|---|---|---|---|
| `cancel_order` | 0.00 | 0.00 | 0.54 | 0.66 | 1.00 | 1.00 |
| `modify_address` | 0.46 | 0.19 | 0.84 | 1.00 | 1.00 | 1.00 |
| `return_items` | 0.00 | 0.00 | 0.91 | 1.00 | 1.00 | 1.00 |
| `lookup_and_report` | 0.32 | 0.01 | 0.10 | 0.70 | 1.00 | 0.84 |
| `refuse_invalid` | 0.00 | 0.00 | 0.00 | 0.87 | 0.74 | 0.99 |
| `exchange_items` *(never in RL)* | 0.01 | 0.01 | 0.59 | 0.55 | 0.83 | 0.38 |

Per-family cells carry a run-to-run standard deviation of about **±0.05** and the headline
about **±0.01**, measured directly by evaluating all six checkpoints twice (§5). Differences
smaller than that are noise, which is why the `exchange_items` regression under long training
is called out below and the `cancel_order` gap between the two 30-step arms is not.

Every episode is also bucketed by the **first** thing that went wrong, so the columns are
mutually exclusive and sum to 100% (`scripts/error_taxonomy.py`):

| Failure mode | Base | + SFT | RL only (30) | SFT+RL (30) | RL only (200) | SFT+RL (200) |
|---|---|---|---|---|---|---|
| no tool call | 13.8% | 50.4% | – | – | 0.7% | – |
| malformed / unknown tool | – | 0.2% | – | 0.4% | – | – |
| illegal write | 4.6% | 0.0% | 18.5% | 2.3% | 6.5% | 2.8% |
| stopped early | 68.2% | 39.3% | 12.0% | 6.5% | – | 0.9% |
| ran out of turns | – | 0.0% | – | 0.2% | – | 0.1% |
| acted, reported badly | – | 6.6% | 14.2% | 10.2% | – | 9.5% |
| reported, acted badly | 0.1% | – | 5.7% | 0.1% | – | – |
| other | – | – | – | 0.5% | – | – |
| **solved** | **13.2%** | **3.5%** | **49.6%** | **79.8%** | **92.9%** | **86.8%** |

Bucket meanings: **no tool call** answered from the prompt alone; **illegal write** called a
write the policy or order state forbids; **stopped early** answered with oracle actions
outstanding; **acted, reported badly** reached the right state without telling the user the
required facts; **reported, acted badly** said the right thing without making it true.

**The highest score in these tables belongs to the worst agent.** RL only (200) reaches 0.929
by discovering that the policy is unscored: it skips user identification in 98.4% of write
episodes and fires the write directly using the id leaked in the prompt. §3.3 is that story.
Read `looked up before writing` alongside `pass^1` throughout — neither is interpretable
alone, which is the main methodological lesson of this run.

`return_items` reads 1.00 for four arms because 7 of its 100 seeds used to be **unsatisfiable**
and are now fixed; §3.4 is that bug, and it is the reason these numbers come from a second
evaluation of every checkpoint rather than the original sweep.

### 3.1 SFT made the model worse — and it was still worth running

SFT dropped success from 0.132 to **0.035**. Half its episodes never call a tool — "used a
tool at all" falls from 0.86 to 0.50 — and the failure transcripts show **two** distinct
causes, both train/serve mismatches inherited from APIGen-MT.

The first is asking instead of acting, in 12 of 16 sampled no-tool episodes:

> *"To assist you with returning the Running Shoes (size 9)… Please provide your user ID so I
> can proceed."*

That is **correct in APIGen-MT and fatal here.** Those trajectories are collected against a
user simulator that answers follow-ups, so requesting missing information is rewarded.
`tau-retail-lite` has no simulator: the instruction is one turn and already contains the email
`find_user_id_by_email` needs, so asking ends the episode having done nothing. The base model
never does this.

The second is subtler and only visible in transcripts — the remaining 4 emit a *corrupted tag*:

```
<function-call>
{"name": "find_user_id_by_email", "arguments": {"email": "sofia.muller23@example.com"}}
</tool_call>
```

The intent, the tool and the arguments are all correct; the opening tag is not the one Qwen3's
template uses, and the closing tag does not match it. The harness sees no tool call and the
episode ends. So part of what looks like "SFT taught the model not to act" is really "SFT
taught the model a different syntax" — a second train/serve mismatch, in the wire format rather
than the dialogue policy. Neither is an optimisation failure, and neither is visible in a loss
curve or an aggregate.

**Judged as a policy, SFT should be deleted. Judged as an initialisation, it is worth +0.301**
— at matched compute RL reaches 0.496 from scratch and 0.797 from the SFT adapter. Only the
ablation separates those claims.

The mechanism is that **the two stages teach opposite, complementary things.** RL-only learns
to act, and acts well where acting is the answer (`return_items` 0.91, `modify_address` 0.84).
What it never learns is restraint: `refuse_invalid` collapses to **0.00**, and
`lookup_and_report`, where the correct action is to touch nothing, drops to **0.10**. It acts
indiscriminately because the environment mostly pays for acting. That is the axis SFT supplies
— it is over-cautious, asking instead of committing, and its illegal writes fall essentially to
zero (0.047 → 0.000). Composed, the two failure modes cancel.

### 3.2 GRPO repaired the failure the diagnosis named

GRPO took 0.035 → **0.797**, fixing precisely what the taxonomy identified: "no tool call" to
0.0%, "stopped early" from 39.3% to 6.5%, tool calls per episode from 1.32 to 3.52. The model
chains, and both SFT pathologies disappear — it acts instead of asking, and it emits the tag
the harness expects. This is the case for online RL in one number: the environment scores what
actually happens, so behaviour rewarded by the SFT corpus but useless at deployment gets
unlearned, which more imitation on that corpus could not achieve.

Longer training (200 steps) sharpened the in-distribution picture and confirmed the
undertraining diagnosis exactly where it was specific: `cancel_order`, whose failures had all
stopped one call short of the write, went to a clean **1.00**, and overall success rose to
0.868. But the held-out family **regressed, 0.55 → 0.38** — a 0.17 drop against a ±0.05 noise
band, so this one is real — while every trained family improved or held. More steps bought
in-distribution accuracy by narrowing the policy. The lever that looks cheapest, train longer,
degrades the property that made the result credible.

### 3.3 The highest score is a reward hack

Removing the SFT prior and training 200 steps produces the best number in the table, 0.929,
and the worst agent. It calls the write tool as its first and only action:

```
[user]      ...cancel an order. My email is yusuf.muller93@example.com... The order id is #W3001.
[assistant] <tool_call>{"name": "cancel_pending_order",
                        "arguments": {"order_id": "#W3001", "reason": "ordered by mistake"}}</tool_call>
[tool]      {"order_id": "#W3001", "status": "cancelled", "refund": 248.4}
```

One call produces the correct final state, and the write tool returns `refund: 248.4`, which
satisfies the output check for free. Lookup compliance falls from 0.936 at 30 steps to
**0.016** at 200 — the policy's first rule, *"Identify the user before acting"*, is skipped in
98.4% of write episodes.

**Two of my design flaws combined to make this optimal, and both are mine, not the model's.**
The `easy` instruction leaks every identifier needed — *"The order id is #W3001, current item
id P1003-0, and I want item id P1003-1"* — so no lookup is ever necessary. And the reward
scores only final state and reported facts; nothing in it references user identification, so
a rule stated in the system prompt is worth exactly zero. Given enough steps, RL correctly
found that the procedure was unpaid and discarded it. It also generalises *better* (held-out
`exchange_items` 0.83) precisely because "call the write with the id from the prompt" is
simpler and more transferable than the intended procedure.

This reframes SFT a third time: its contribution is not only restraint but **anchoring the
model to a procedure the reward never pays for**. SFT→RL holds 100% compliance at both 30 and
200 steps because it started there and no gradient pushed it off. That is a real benefit and
an accidental one — regularisation toward intent, not something the objective earned.

**So the best genuine agent in this table is SFT+RL at 30 steps (0.797), and everything above
it is measurement artifact.** Reporting 0.929 as the headline would have been the most
flattering and least honest reading of the run.

**Caveat on all RL numbers.** GRPO optimises the same reward the harness scores, so RL
checkpoints are trained on the eval metric in a way base and SFT are not. Disjoint seeds and a
held-out family control for memorising *tasks*, not for objective and metric coinciding.

### 3.4 A benchmark bug the aggregate hid, and the test that should have caught it

In the original sweep, `return_items` read exactly **0.93** in three independently trained
arms. A number that stable across different checkpoints is a property of the benchmark, not
the model.

It was. On 7 of 100 test seeds the task could not be solved truthfully. An order may list the same
`item_id` on more than one line — **7.8% of generated orders do** — and
`return_delivered_order_items` refunds every matching line, while the task asked for a single
unit price:

```
required_outputs : ['65', '65.00']            # one Backpack (grey)
tool returned    : {"refund": 130.0}          # the order lists it twice
```

The agent reports 130.0, which is what the environment told it and what actually happened, and
was scored wrong. 28 of 2,400 episodes (1.2%) were unwinnable, capping `return_items` at 0.93 —
so those three arms had solved **everything solvable**. With the tasks repaired the family
reads a clean **1.00** for four arms, exactly the predicted +0.07, and it contributes no
residual failures at all to §3.5.

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
over the full evaluation split, and fails loudly on the seed above.

The generalisable form: a fixture derived from the thing under test proves nothing. The oracle
test looked like the most load-bearing test in the repo — its own docstring says so — and half
of it was circular.

### 3.5 What is actually left to fix

Decomposing the residual failures of each RL arm by which half of `r_outcome = r_action *
r_output` failed:

| | SFT+RL (30) | SFT+RL (200) | RL only (200) |
|---|---|---|---|
| total failures | 486 (20.2%) | 318 (13.2%) | 171 (7.1%) |
| state right, **report** wrong | 50.2% | 71.7% | – |
| report right, **state** wrong | 40.3% | 15.7% | 62.0% |
| neither | 9.5% | 12.6% | 38.0% |

For the best genuine agent, **the largest share of what remains is reporting, not acting** — it
does the job and then fails to say the required fact. That is a cheaper problem than tool
selection, and it is concentrated: of its 486 failures, 179 are `exchange_items` (the held-out
family), 137 `cancel_order`, 118 `lookup_and_report` and 52 `refuse_invalid`. Longer training
pushes the mix further toward reporting (71.7%), consistent with RL fixing execution first.
If reporting were perfect this checkpoint would score **0.899**; if actions were perfect,
0.879. Reporting is the larger lever.

RL only (200) inverts this completely — **none** of its failures are reporting failures, and
62% are state-wrong-report-right: an agent that says the right thing without reliably making it
true, the same disposition as the §3.3 shortcut.

**A harness hypothesis, tested and wrong.** The largest bucket is `exchange_items` episodes
with `r_progress` 1.0 and `r_output` 0 — executed perfectly, scored zero on reporting. The
output matcher is a substring test that demanded the leading `#`, so *"Your order W3006 has
been exchanged"* failed, and an order id is the entire required output of two families. A real
defect, worth fixing on its own terms, and an attractive explanation — so it was measured
rather than assumed. Ids now accept both renderings (`_order_id_variants`, pinned by a test).
`exchange_items` moved **0.547 → 0.552**, well inside the ±0.05 noise band. The hypothesis is
falsified; the `#` was not the problem.

The transcripts show what is. Asked to *"Confirm the order id when done"*, the model executes
the exchange correctly and then confirms the **item** ids instead, adding a $120.00 "refund"
that an exchange does not produce. A genuine instruction-following failure, not a formatting
artifact — and one that only became visible because the transcript sampler was fixed at the
same time. It had been saving `results[:12]`, which was three families at one seed and all of
them passes: a file whose entire purpose is explaining failures, containing none. The reporting
shaping term is the right lever, and loosening the matcher further would have buried this.

**How the RL target was chosen in the first place.** An aggregate of 0.132 says a checkpoint is
bad, not what to fix. The informative entry for the base model is what is *absent*: malformed
calls and hallucinated tools are **0%**, and **no episode hit the turn limit** — all 2,400 ended
because the model chose to answer. It is not that it cannot spell a tool call; it does not take
enough turns. Of the 2,083 failures that ended in an answer, 86.8% have `r_progress` exactly
`0.0`, so outcomes are near-bimodal — whole task or nothing, the regime where group-relative
advantage collapses. The taxonomy named premature termination, and that is exactly what RL
moved: "stopped early" 68.2% → 6.5%, "no tool call" 13.8% → 0%.

---

### 3.6 BFCL: the external check, and what it cost the story

Everything above is measured on an environment I wrote. BFCL is the uncontaminated test —
fully offline, deterministic AST and state scoring, no judge, no overlap with
`tau-retail-lite`. Four checkpoints were run over 2,340 test cases. Predictions were
**committed before the run** (`BFCL_PREREGISTRATION.md`) so they could not be fitted afterwards.

BFCL V4 `Overall` is deliberately not reported: 40% of its weight is agentic web search and
long-session memory, which this pipeline never trained. A composite dominated by an untrained
skill is uninformative in both directions.

| | Base | + SFT | SFT+RL (30) | RL only (200) |
|---|---|---|---|---|
| `simple_python` (400) | 0.855 | 0.675 | 0.812 | 0.850 |
| `multiple` (200) | 0.850 | 0.505 | 0.840 | 0.840 |
| `parallel` (200) | 0.725 | **0.000** | **0.000** | 0.715 |
| `parallel_multiple` (200) | 0.750 | **0.000** | **0.000** | 0.720 |
| **AST pooled (1,000)** | **0.807** | **0.371** | **0.493** | **0.795** |
| `multi_turn_base` (200) | 0.080 | 0.020 | 0.070 | 0.065 |

**No trained arm beats base on anything.** The in-domain story was 0.132 → 0.797, a six-fold
gain. Externally that gain is worth nothing, and the SFT arms are catastrophically worse.

Worth being precise about what "off-distribution" means here, because the two stages use
different data but not different domains. SFT is APIGen-MT-5k, RL is `tau-retail-lite`, and the
former was chosen *for* its domain overlap with the latter — both are tau-bench retail/airline
back-office work. Nothing in this pipeline ever trained on diverse tools, so BFCL is testing
transfer out of a single narrow domain family, not merely out of one environment.

**SFT destroyed parallel calling outright, and RL did not bring it back.** 0 of 200, twice, in
two independently trained arms. Per §3.4's own lesson, a number that stable across checkpoints
is usually a property of the benchmark — so I read the failure records rather than assume. It is
real, and the mechanism is exact. The syntax is perfect; the *count* is wrong:

```
error      : ["Wrong number of functions."]
raw output : <tool_call>{"name": "spotify.play",
                         "arguments": {"artist": "Taylor Swift", "duration": 20}}</tool_call>
```

The task asked for Taylor Swift *and* Maroon 5. **The model was never shown a single example of
two calls in one turn.** Across the SFT corpus there are 16,732 assistant messages containing
tool calls and *none* of them holds more than one — and that is structural rather than
incidental, since `prepare_sft.py` maps each ShareGPT `function_call` turn onto its own assistant
message, so `tool_calls` is always a one-element list. `tau-retail-lite` is one-call-per-turn
too. So neither stage ever demonstrated or rewarded a second call in the same turn, the model
learned "emit one call and stop" as a hard rule, and a capability the base model had was trained
out of it. **This is the clearest evidence in the project that narrow training
data removes capabilities that no in-domain metric can see** — `tau-retail-lite` cannot even
express the task that fails.

**On-policy RL preserved general capability where imitation destroyed it.** RL-only sits at
0.795 AST against base's 0.807 — statistically indistinguishable — despite training 200 steps in
the same single-call environment that ruined the SFT arms. SFT saw 1,000 trajectories and lost
0.436. The plausible mechanism is that on-policy RL can only reweight behaviour the policy
already produces, so capability outside the training distribution is never pushed on, whereas
imitation actively overwrites the output distribution toward the corpus. SFT+RL partially
repairs the damage (0.371 → 0.493, `multiple` 0.505 → 0.840) but cannot recover what collapsed
to zero.

**The one axis that transfers cleanly is the act/abstain trade-off — and it confirms §3.1
externally.** Splitting the relevance categories into "should not call" (1,124 cases) and
"should call" (16) shows a single monotonic axis, not a competence difference:

| | should NOT call | should call |
|---|---|---|
| Base | 0.801 | 0.625 |
| + SFT | **0.903** | **0.375** |
| SFT+RL (30) | 0.634 | 0.750 |
| RL only (200) | 0.738 | **0.812** |

SFT is the most abstemious model and the worst at acting when acting is correct; the RL arms are
the reverse. That is exactly the in-domain diagnosis — SFT asks instead of acting, RL acts
indiscriminately — reproduced on a benchmark I did not write. Note that the pooled "restraint"
number is meaningless on its own: with 1,124 abstain cases against 16 call cases, a model that
simply stopped calling functions would top the group while being useless. SFT scoring highest
there is the artifact; the split is the finding.

**P1, the load-bearing prediction, is unresolved.** §3.3 claims RL-only's in-domain lead is a
reward hack, predicting it should lose to SFT+RL externally. Multi-turn: 14/200 against 13/200,
p=0.84. The direction matches and the magnitude is one test case. **That is not support, and it
is reported as unresolved rather than dressed up as confirmation.** The honest reading is that
BFCL multi-turn at n=200 and ~6% accuracy has nowhere near the power to test it; resolving it
needs all four multi-turn categories, roughly 8 GPU-hours.

Of the four pre-registered predictions: **P2 confirmed** far more strongly than predicted (SFT
damage is real, external, and catastrophic rather than marginal), **P3 confirmed** (all arms
under 0.15 multi-turn), **P1 unresolved** for lack of power, and **P4 refuted** — I predicted the
RL-only arms would show *weaker* restraint than SFT-initialised ones, and RL-only is in fact
better at abstaining than SFT+RL (0.738 vs 0.634).

**What this does to the headline.** 0.797 remains a real measurement of a real capability
gain *in the environment it was trained for*. What BFCL establishes is that the gain is
environment-specific, and that it was bought with general capability I did not know I was
spending. A result reported as "SFT → RL took a 0.6B model from 0.13 to 0.80 on multi-turn tool
use" would have been true and badly misleading. That is the single strongest argument in this
project for running an external benchmark you did not write.

### 3.7 Fixing the parallel collapse, and what it cost

§3.6 offered a narrow explanation for `parallel` reading 0/200: the model was never shown two
calls in one turn. The duller competing explanation is that LoRA SFT at this scale degrades the
model broadly and parallel calling is simply what breaks first. Those predict different things,
so the corpus was changed and nothing else.

`sft_mixed` trains on 500 APIGen-MT trajectories plus 500 from
NousResearch/hermes-function-calling-v1, which supplies exactly what APIGen lacks — 56.8% of its
tool-calling turns carry more than one call, across 35 domains rather than one. Total
trajectories stay at 1,000, matching the `sft` arm, so composition is the only variable. xLAM was
the first choice and is gated; ToolACE encodes calls as a bracketed DSL with spaces in function
names. Hermes needs neither a token nor a bespoke parser, and already uses the `<tool_call>` JSON
Qwen3's template emits. Predictions were committed before training (`BFCL_PREREGISTRATION.md`).

| | Base | + SFT | **SFT mixed** |
|---|---|---|---|
| `simple_python` | 0.855 | 0.675 | 0.792 |
| `multiple` | 0.850 | 0.505 | 0.760 |
| `parallel` | 0.725 | **0.000** | **0.705** |
| `parallel_multiple` | 0.750 | **0.000** | **0.590** |
| **AST pooled** | **0.807** | **0.371** | **0.728** |

**The explanation holds.** `parallel` goes 0.000 → 0.705 against base's 0.725, and pooled AST
recovers 0.357 of the 0.436 that SFT destroyed — about 82% of the damage, from changing nothing
but which trajectories the model read. The failure was never capacity or general forgetting; it
was a missing demonstration. 284 multi-call trajectories, 12.9% of the corpus's tool-calling
turns, were enough. **P5 and P6 confirmed.**

**The bug that nearly buried it.** All 284 of those trajectories were silently discarded by my
own masker. Qwen3's template merges consecutive tool messages into a single user turn, so
rendering after each response is not append-only, the prefix check fails, and `build_example`
returns `None`. It could only ever bite on multi-call turns, which no previous corpus contained —
so the mixed arm would have trained on *zero* parallel examples while reporting a full 1,000, and
the tidy conclusion would have been "adding parallel data does not help, so the damage is
broad." Nothing downstream would have shown it: the loss curve is unremarkable and the arm simply
fails to learn the one thing it exists to learn. It was caught by counting how many multi-call
trajectories survived `build_example` before spending any GPU time, and is now pinned by
`test_a_turn_calling_two_tools_at_once_survives_masking`.

**It was not free.** The repair bought AST accuracy by moving the model along the same act/abstain
axis as §3.6, and the trade is visible in the decomposition:

| | should NOT call | should call |
|---|---|---|
| Base | 0.801 | 0.625 |
| + SFT | **0.903** | **0.375** |
| SFT mixed | 0.642 | **0.812** |

SFT was the most abstemious checkpoint in the project; the mix turns it into one of the most
eager, best-in-class at calling when calling is right (0.812) and materially worse than base at
staying silent (0.642 against 0.801). That is unsurprising in hindsight and was *not* predicted:
every Hermes trajectory calls a function, so a corpus half made of them teaches "call something".
It also means the arm cannot be read as strictly better — it is better where the old one was
catastrophic and worse where the old one was strong.

Multi-turn stayed poor (0.025 against base's 0.080) and in-domain `pass^1` moved only 0.035 →
0.055, both consistent with **P7**: the mix does nothing about the actual in-domain failure, which
is asking a user simulator that does not exist.

**What this does and does not license.** It licenses a specific claim — narrow SFT data removed a
capability, and restoring the demonstrations restored the capability. It does not show the mix
makes a better agent, because SFT's value here was never as a policy but as an initialisation
(+0.301, §3.1), and the mix changes the corpus that produced it. §3.8 tests that directly.

### 3.8 The repair does not pay for itself

The only reason to run SFT in this pipeline was §3.1: as a policy it is worse than base (0.035
against 0.132), but as an RL initialisation it is worth +0.301 at matched compute. §3.7 replaced
half the corpus that produced that number, so it has to be re-measured rather than assumed.
`grpo_mixed` runs 30 GRPO steps from `sft_mixed`, matching the `grpo` arm step for step, so the
prior is the only variable. Predictions were committed first (P8–P10).

| in-domain, 2,400 episodes | RL only (30) | SFT + RL (30) | **SFT mixed + RL (30)** |
|---|---|---|---|
| **pass^1** | 0.496 | **0.797** | **0.475** |
| `return_items` | 0.91 | 1.00 | **0.16** |
| `cancel_order` | 0.54 | 0.66 | **0.23** |
| `modify_address` | 0.84 | 1.00 | 0.86 |
| `refuse_invalid` | 0.00 | 0.87 | 0.77 |
| looked up before writing | 0.94 | 1.00 | **1.00** |

**The prior is worth nothing.** 0.475 against 0.496 from no prior at all: the entire +0.301
evaporated. **P8 refuted.** The damage is concentrated in the write-heavy families —
`return_items` collapses from 1.00 to 0.16, below even the arm that never saw SFT — which is
where retail protocol matters rather than general tool syntax, and precisely what the 500 removed
APIGen trajectories carried. So SFT's value as an initialisation was never "SFT" in the abstract;
it was *domain-matched* trajectories, and half of them is not half as good.

That looks like it reframes §3.7 as a dilemma — parallel calling and the RL prior trading directly,
0.357 of pooled AST bought with 0.322 of in-domain `pass^1`. **§3.11 shows that reading is wrong,
and the trade was an artifact of my own control.** The 1,000-trajectory cap existed so composition
would be the only variable; under it, adding Hermes necessarily removes the APIGen that carries
retail protocol. Lift the cap and both goals are available at once.

**Anchoring survived, which was not expected.** P9 predicted lookup compliance would fall, on the
theory that §3.3's perfect compliance was a side effect of SFT's timidity — and `sft_mixed` traded
away exactly that timidity. Compliance is 0.998. **P9 refuted**, in the useful direction: the
checkpoint that follows the lookup-then-write procedure essentially always is simultaneously the
most eager one in the project (should-not-call 0.454). Procedure-following and eagerness are
independent axes, so whatever SFT installs that resists the §3.3 reward hack is not reluctance,
and it is robust to changing half the data.

**On BFCL the repair holds, and restraint keeps sliding.**

| | Base | + SFT | SFT+RL (30) | SFT mixed | **SFT mixed + RL (30)** |
|---|---|---|---|---|---|
| `parallel` | 0.725 | 0.000 | 0.000 | 0.705 | **0.705** |
| `parallel_multiple` | 0.750 | 0.000 | 0.000 | 0.590 | 0.605 |
| **AST pooled** | 0.807 | 0.371 | 0.493 | 0.728 | **0.755** |
| **Restraint pooled** | 0.798 | **0.896** | 0.636 | 0.645 | **0.460** |
| should NOT call | 0.801 | 0.903 | 0.634 | 0.642 | **0.454** |
| should call | 0.625 | 0.375 | 0.750 | 0.812 | **0.875** |

**P10 confirmed**: thirty steps of single-call RL do not erase parallel calling once the prior has
it. The §3.7 fix belongs before RL and stays put, which is what makes it worth having at all — a
repair the second stage undid would be useless here.

The restraint column is the finding I did not predict at any point. `sft_mixed + RL` is the most
capable arm at *calling* (should-call 0.875, best measured, above base's 0.625) and the worst at
*not* calling (0.454 against base's 0.801). Both stages push the same direction: Hermes teaches
"call something" because every one of its trajectories does. GRPO's contribution is more specific,
and I got it wrong at first.

Ordering the six arms by should-not-call recovers the should-call column almost exactly in reverse
— 0.375, 0.625, 0.812, 0.812, 0.750, 0.875, one inversion, and that between two arms 0.008 apart on
the sort key. Every intervention up to this point turned that dial, and none of them deliberately.
The caveat is that `should call` is BFCL's `live_relevance`, only 16 cases, so each step is one or
two examples and it cannot carry the claim alone; the abstention column (n=1,124) is what makes the
pattern solid. §3.10 then shows the two are separable after all — the ordering described six arms
that happened to vary one thing, rather than a constraint.

### 3.9 The RL environment trains against abstention, and a bug fix is why

My first explanation for the restraint column was that nothing in the pipeline pays for abstaining.
That is false, and checking it against the environment code rather than my own summary of it
produced the sharpest finding in the project.

The environment has a `refuse_invalid` family — the user asks to cancel an already-delivered order,
which the policy forbids — and it pays *full* reward for declining, up to the same 1.3 any other
family can earn. So abstention is rewarded. The catch is how it must be expressed:

```181:190:src/tooluse/env/tasks.py
    elif family == "refuse_invalid":
        order_id, order = _pick_order(rng, db, DELIVERED)
        user = db["users"][order["user_id"]]
        # The user asks for something policy forbids: cancelling an already-delivered order.
        oracle = [{"name": "transfer_to_human", "args": {}}]
```

`transfer_to_human` is in `WRITE_ACTIONS`, and `r_action` compares the final database hash against
an oracle database in which `escalated=True`. A model that simply says "I can't help with that"
and calls nothing leaves the database untouched, fails the state check, and scores **exactly
0.0**. In the committed episode records, across 800 refusal episodes in the two RL arms, **no
successful episode made zero tool calls**; successful ones average **4.47** calls (`grpo`) and
**4.57** (`grpo_mixed`), typically lookups, then a doomed `cancel_pending_order` attempt costing
−0.4, then the escalation. `refuse_invalid` is 1 of 5 RL training families.

So the environment does not fail to pay for silence — **it trains, in roughly a fifth of RL
samples, that the correct response to an out-of-policy request is four to five tool calls.** BFCL
irrelevance asks for precisely the opposite: no call, decline in text. The conflict is not an
oversight in the reward, it is an actively trained-in opposite, which is a much better fit for the
data than my original story: restraint degrades monotonically with RL exposure (0.896 → 0.645 →
0.460) while lookup compliance stays pinned at 0.998, because the environment is specific about
*which* procedure to follow and specific about always having one.

**And it is there because of a reward-hack patch.** The test that pins this behaviour says why:

```67:72:tests/test_harness.py
def test_refusal_requires_escalating_not_just_apologising() -> None:
    """Regression: a model that only says "I can't help" must not score on a refusal task.

    Before `transfer_to_human` had a state footprint, doing nothing produced the correct
    final database and the task had no required outputs, so passivity scored a perfect 1.0.
    """
```

The original bug was the mirror image: with no state footprint and no required outputs, the
*correct* final state for a refusal task was the untouched database, so a model that did nothing at
all scored a perfect 1.0 on the family meant to test judgement. Giving escalation a state footprint
fixed that cleanly and, in doing so, made "call nothing" unscoreable anywhere in the environment.
**A patch for one reward hack installed the bias that an external benchmark measured two weeks
later**, and neither the in-domain metric nor the test suite could see it, because both were built
around the same assumption that a correct episode has a state footprint.

The narrow fix is not to weaken that test. Escalating *is* correct retail conduct and the policy
says so, so `refuse_invalid` should keep requiring it. What is missing is a family where the right
answer genuinely is no call at all, scored by the output check rather than the state check — the
machinery for which already exists, since `lookup_and_report` is scored with an empty oracle and
`r_progress` defined as having refrained from writing. It needs one extension: for such a family,
refraining must mean *no calls*, not merely no writes.

### 3.10 Training restraint back: it works, and it barely transfers

`grpo_abstain` adds one family, `irrelevant_request`, in which the user asks something no tool here
can serve and the correct episode makes no call at all. Two design points earn their keep. Restraint
gates the **state** term rather than progress, because reads leave the database untouched and the
hash check alone cannot tell a refusal from four wasted lookups followed by one — as progress-only
scoring, "read four times then decline politely" would have scored a perfect 1.0 on the family built
to teach restraint. And topics are **held out** between train and test the way databases already
are, so the in-domain number cannot be earned by memorising 24 strings. GRPO then runs 30 steps from
`sft_mixed`, matching `grpo_mixed` exactly; the system prompt is untouched and the headline 600-task
split fingerprints identically, so every earlier number stays comparable. P11–P14 were committed
first.

| | SFT mixed + RL (30) | **+ abstention family** | Base |
|---|---|---|---|
| in-domain `pass^1` (6 families) | 0.475 | 0.436 | 0.132 |
| in-domain abstention, held-out topics | *(not expressible)* | **0.820** | — |
| ...of which made zero calls | — | **400/400** | — |
| BFCL restraint pooled | 0.460 | **0.545** | 0.798 |
| BFCL should NOT call | 0.454 | **0.541** | 0.801 |
| BFCL should call | 0.875 | 0.812 | 0.625 |
| BFCL AST pooled | 0.755 | 0.747 | 0.807 |
| `refuse_invalid` | 0.770 | 0.723 | 0.000 |

**All four predictions confirmed, and the headline is that it works.** BFCL restraint goes 0.460 →
0.545 (z = 4.06), against a measured noise floor of 0.012–0.017, while AST holds at 0.747 against
0.755 — a difference smaller than the floor. **P11 and P12 confirmed.** In-domain `pass^1` costs
0.039 and `refuse_invalid` 0.047, both inside the predicted bands (**P13, P14**). The model did not
simply go quiet either: it still calls a tool in 100% of episodes on the six normal families.

**So the act/abstain axis is not one dial after all.** §3.8 read the near-perfect reverse ordering
of the two columns as a single parameter every intervention happened to turn. This arm moves
abstention by +0.087 while leaving syntax alone and giving back one test case of sixteen on
should-call, which is the one result in the project that separates them. The reverse ordering was
real but it was a description of six arms that all happened to vary one thing, not a constraint.
Restraint is separately addressable, and the earlier framing was too fatalistic.

**The finding that matters more is the size of the transfer gap.** In-domain the intervention is
not merely successful, it is *saturated*: **400 of 400** held-out-topic episodes make zero tool
calls, perfect restraint on topics never trained on. The same checkpoint recovers **25%** of the
distance to base on BFCL's should-not-call (0.454 → 0.541 against 0.801). Perfect in-domain
generalisation across topics buys a quarter of the external gap. Whatever `irrelevant_request`
teaches — "requests with no email and no order id get no tool call" — is narrower than "decide
whether the tools you were handed can serve this request", which is what BFCL grades. The residual
0.180 in-domain is entirely the text check (`r_action` is 1.000, `r_output` 0.820): the model
abstains and sometimes forgets to say why.

That is the same lesson as §3.6 arriving from the opposite direction. There, an in-domain metric
could not see a capability being destroyed; here, an in-domain metric is saturated by an
intervention that moves the external one a quarter as far. **Both times the environment was too
narrow to measure what I wanted to claim, and both times only the external benchmark showed it.**
The fix is not a better reward term but a wider distribution of irrelevance — varied tool
inventories, requests that are plausible-but-unserveable rather than obviously off-topic — which is
a data problem, not a scoring one.

### 3.11 The dilemma was my control talking

§3.8 concluded that repairing parallel calling costs the whole of SFT's value as an RL prior, and
§3.10 sat on top of that as a second trade. Both rest on a comparison in which the corpus total was
pinned at 1,000 trajectories — deliberately, so that composition would be the only variable. But a
control is not a budget. Under a fixed total, adding 500 Hermes trajectories *necessarily* removes
500 APIGen ones, and §3.8 established that the removed APIGen is precisely what carries retail
protocol. The dilemma might be a property of the models, or it might be a property of my
experimental design, and those are distinguishable for twenty minutes of GPU.

`sft_1500` keeps all 1,000 APIGen trajectories and *adds* 500 Hermes on top. `grpo_1500` then runs
30 GRPO steps from it, matching `grpo` and `grpo_mixed` step for step.

| | RL only (30) | SFT+RL (30) | SFT mixed + RL | **SFT 1500 + RL** |
|---|---|---|---|---|
| **in-domain `pass^1`** | 0.496 | 0.797 | 0.475 | **0.880** |
| looked up before writing | 0.94 | 1.00 | 1.00 | **1.00** |
| `cancel_order` | 0.54 | 0.66 | 0.23 | **0.98** |
| `return_items` | 0.91 | 1.00 | 0.16 | **1.00** |
| `refuse_invalid` | 0.00 | 0.87 | 0.77 | **0.96** |
| `exchange_items` *(held out from RL)* | 0.59 | 0.55 | 0.23 | 0.48 |
| BFCL `parallel` | 0.715 | **0.000** | 0.705 | **0.695** |
| BFCL AST pooled | 0.795 | 0.493 | 0.755 | **0.759** |

**There was never a trade.** `grpo_1500` reaches **0.880**, above the 0.797 that the substitution
was supposed to have cost, while keeping `parallel` at 0.695 and pooled AST at 0.759 — the best AST
of any GRPO arm. **P15 and P16 confirmed**, P15 more strongly than predicted (I said 0.65–0.80). It
is also not a reward hack: lookup compliance is 1.000, and it beats `grpo` on five of six families,
including `cancel_order` 0.66 → 0.98 and `refuse_invalid` 0.87 → 0.96. At 30 steps it edges out
`grpo_long`'s 0.868 at 200. **This is the best genuine agent in the project.**

The one place it does not win is `exchange_items`, the family held out of RL entirely: 0.48 against
`grpo`'s 0.55. Worth naming rather than burying, since that is the family that measures whether RL
improved the protocol generally or only the families it was paid for.

**What this costs is somewhere else entirely, and it is severe.**

| | Base | SFT+RL (30) | SFT mixed + RL | + abstention | **SFT 1500 + RL** |
|---|---|---|---|---|---|
| **BFCL restraint pooled** | 0.798 | 0.636 | 0.460 | 0.545 | **0.235** |
| should NOT call | 0.801 | 0.634 | 0.454 | 0.541 | **0.225** |
| should call | 0.625 | 0.750 | 0.875 | 0.812 | **0.938** |

`grpo_1500` is the most extreme point on the act/abstain axis measured anywhere in this project:
best-in-class at calling when calling is right (0.938, above base's 0.625) and **less than a third
as good as base at staying silent** (0.225 against 0.801). Restraint is half what the capped mix
achieved, which was already the worst arm at the time. **P17 refuted** — I predicted the larger
APIGen share would pull *back* toward abstention, since APIGen is the abstemious source, and
`sft_1500` instead abstains worse than `sft_mixed` (0.564 against 0.642) before RL halves it again.

So the honest correction is two-sided. §3.7's repair and §3.8's prior are not in tension and I
should not have said they were; that was my control, not the model. But the act/abstain axis is a
genuine constraint that every single intervention in this project has pushed the same way, and the
best in-domain agent is the worst abstainer. §3.10 showed that axis is separately addressable — the
abstention family bought +0.087 of restraint for 0.039 of `pass^1` — so the run that is actually
worth doing next is `sft_1500` plus the abstention family together. That combination is untested,
and it is the only configuration in which the project's best agent might also be a safe one.

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
- **1.2% of evaluation episodes were unwinnable** (§3.4). Fixed, pinned by a test, and all six
  arms re-evaluated on the corrected metric, so every number here is comparable — but that
  correction was found only because one family sat at a suspiciously stable 0.93. I have no
  guarantee a similar defect is not still present in a family whose numbers look unremarkable.
- **The headline benchmark is one I wrote.** Train and test share no database and one task
  family is held out of RL entirely, which controls for memorisation but not for the
  environment being easier, or differently shaped, than a real benchmark.
- **The gains do not transfer, and BFCL now says so** (§3.6). No trained arm beats base on any
  external category, and the SFT arms lose 0.436 AST accuracy — including parallel calling
  falling to exactly zero. The in-domain result stands as a measurement of the environment it
  was trained for and nothing wider.
- **I reported a trade that did not exist, because I never questioned my own control** (§3.8 vs
  §3.11). Holding the corpus at 1,000 trajectories made repairing parallel calling look like it
  cost the whole of SFT's value as an RL prior. Lifting the cap gives 0.880 — better than the 0.797
  supposedly lost — with parallel calling intact. The result was wrong for a full iteration, and
  nothing in the data flagged it: the numbers were correct, the comparison was sound, and the
  conclusion drawn from it was still an artifact of a design decision I had stopped seeing.
- **The best agent is the worst abstainer, and that trade is real** (§3.11). `grpo_1500` reaches
  0.880 in-domain and 0.225 on BFCL should-not-call, against base's 0.801. Every intervention in
  this project pushed the same way on that axis. §3.10 shows it is separately addressable, but the
  combination that would test it — `sft_1500` plus the abstention family — was not run.
- **The RL environment actively trains against abstention** (§3.9). Ordering the arms by
  should-not-call recovers should-call in near-reverse, and the final arm is the worst abstainer
  measured (0.454 against base's 0.801). My first explanation — that nothing pays for abstention —
  was wrong: `refuse_invalid` pays full reward for declining, but only via `transfer_to_human`,
  a write, so a text-only decline scores 0.0 and successful refusals average 4.5 calls. A fifth of
  RL training therefore teaches that out-of-policy requests warrant four to five calls, which is
  the opposite of what BFCL irrelevance scores. Worse, that requirement exists because it patched
  an earlier hack in which passivity scored a perfect 1.0.
- **The fix for that is saturated in-domain and only a quarter effective externally** (§3.10).
  400/400 held-out-topic episodes abstain correctly, yet BFCL should-not-call recovers 25% of the
  distance to base. The in-domain family teaches a narrower rule than the benchmark grades, and no
  amount of in-domain measurement would have revealed the difference — the same blind spot as §3.6,
  approached from the opposite direction.
- **BFCL multi-turn was underpowered for the question it was run to answer.** One category,
  200 cases, ~6% accuracy: the 95% interval is roughly ±3.3 points, so P1 needed a gap of five
  points to register and got one test case. The four-category multi-turn suite (800 cases) was
  priced at about 8 GPU-hours and not bought. The result is reported as unresolved.
- **A merge bug silently invalidated the first BFCL sweep.** All four "merged" checkpoints were
  byte-identical copies of base, so the first run scored the base model four times and produced
  a tidy, entirely false "nothing transfers, all arms identical" conclusion. It was caught only
  by hashing the weights (`verify_merged_differ`). The accident did leave something useful: four
  identical models scored independently measures BFCL's own run-to-run spread at **0.012–0.017**,
  which is the floor any claimed difference has to clear.
- **RL is trained on the eval metric.** GRPO optimises the same grounded reward the harness
  scores. Disjoint seeds and a held-out family rule out memorising tasks, but not the
  objective and the metric being the same function. Base and SFT get no such advantage, so
  the 0.132 → 0.797 comparison is not between equals.
- **Evaluation noise is now measured; training noise is not.** Every checkpoint was evaluated
  twice (the §3.4 fix forced a re-baseline), which gives a direct estimate: **±0.01 on the
  headline, ±0.05 per family** (mean |delta| 0.005 and 0.016, max 0.068). Differences smaller
  than that are not real, and I have tried to avoid claiming any. This says nothing about
  *training* variance — each arm was trained once, so the SFT-as-prior result (+0.301) rests
  on a single pair of runs.
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
   that would let the 0.797 be described as tool use rather than as this environment.
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
