# Multi-turn tool use on a 0.6B model: SFT → GRPO → eval

**Time budget:** 4 hours, hard cap. **Compute:** Modal, single A10.
**Model:** `Qwen3-0.6B`. **Artifacts:** `results/`, one JSON record per episode.

Task success on 2,400 held-out episodes per checkpoint:

| | Base | + SFT | + SFT + GRPO |
|---|---|---|---|
| **pass^1** | 0.140 | 0.037 | **0.794** |

The headline is not the final number but its shape. **SFT made the model measurably worse**,
and for a diagnosable reason: APIGen-MT trains it to ask a user simulator for missing
information, and this environment has no simulator, so asking ends the episode having done
nothing. **GRPO then recovered the loss and went well past the base model**, because online
RL optimises what actually happens at deployment rather than an imitation target. A family
excluded from RL training entirely rose from 0.01 to 0.55, so the gain is a reusable
procedure rather than memorised scripts.

That sequence — a stage that hurt, diagnosed rather than hidden, then corrected by the next
stage — is the argument for this pipeline. The one number that moved the wrong way (policy
violations tripled under RL) is reported in §3 rather than omitted.

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
stated plainly in §6 — the headline number is measured on a benchmark I wrote.

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

The product is what makes the reward hard to game from either side. State-only would reward
an agent that mutates the database and says nothing useful; output-only would reward one
that recites a plausible refund and never issues it.

The ground truth is **derived**, not written by hand: the task sampler executes the task's
oracle action list against a copy of the initial database and hashes the result. The oracle
and the reward therefore cannot drift apart. `tests/test_env.py` asserts the oracle scores
1.0 on every family across 40 seeds; if that ever fails, the task is unsolvable and any RL
number measured on it would be meaningless.

### Shaping, and why each term cannot be farmed

Binary outcome alone is far too sparse for a 0.6B. Total reward adds:

| Term | Range | Purpose | Why it is not exploitable |
|---|---|---|---|
| `r_outcome` | {0,1} | the real objective | — |
| `r_progress` | [0,1] | fraction of oracle actions performed **with correct arguments** | matched greedily against the oracle, so repeating one correct call cannot substitute for a missing one; a wrong write earns nothing and costs a violation |
| `r_format` (protocol) | [0,1] | parseable call, existing tool, sane arguments | annealed toward zero as it saturates, so it stops competing with the objective |
| `p_efficiency` | ≤0 | penalises redundant calls | **only charged when the episode already succeeded**, so it can never outrank correctness |
| `p_violation` | ≤0 | illegal writes, hallucinated tools, malformed calls | always charged |

### The reward-hacking hole the tests caught

The first version of `refuse_invalid` was broken in a way that is easy to miss and that a
loss curve would never reveal. The correct final state for a refusal is "database
unchanged", and the family required no output. So an agent that replied *"I'm sorry, I
can't help with that"* and did nothing scored `r_action = 1`, `r_output = 1` (vacuously),
and a **perfect 1.0**. The family intended to punish over-eager writing was in fact paying
full marks for passivity — and, worse, teaching it.

The fix was to give escalation a state footprint: `transfer_to_human` now sets an
`escalated` flag that is part of the hashed state, so refusing *without* escalating fails
the state check. Only the fact of escalation is recorded, never the agent's free-text
summary, so wording cannot decide the reward.
`tests/test_harness.py::test_refusal_requires_escalating_not_just_apologising` pins it.

### Handling sparse signal

GRPO's advantage is group-relative: when all G rollouts for a task score the same, the
advantage is zero and that task contributes nothing. At 0.6B the default is "all G fail".
Four things address this, and one metric monitors it:

1. **The shaping above**, so groups differentiate on protocol and partial progress long
   before any full success appears.
2. **An easy difficulty**: every task states the order id, keeping the oracle to 1–3 calls.
3. **G = 8**, to raise the chance of at least one success per group.
4. **`frac_reward_zero_std`**, which TRL logs, measures this failure directly and is the
   metric to watch for a longer run (§7).

The assumption behind (3) was that SFT would be load-bearing — that its job was lifting the
success rate off the floor so RL had variance to exploit. §3 shows that assumption was
wrong in an instructive way. SFT *lowered* the success rate to 0.037, and GRPO still reached
0.794 from there. The shaping terms, not the SFT prior, are what supplied the early
gradient: a model that asks a question scores zero on progress and format alike, while one
that emits a well-formed call scores on both, so groups differentiated even when no rollout
in them succeeded outright.

---

## 3. Results

All three checkpoints were evaluated with identical decoding (T=0.7, top-p 0.95, fixed
seed), identical prompts and identical chat-template settings, on **600 held-out tasks × 4
trials = 2,400 episodes each**. Test seeds (100000–100099) are disjoint from training seeds
(0–1999), so no evaluation task shares a database with any training task.

| Metric | Base | + SFT | + SFT + GRPO |
|---|---|---|---|
| **pass^1** | 0.140 | 0.037 | **0.794** |
| **pass^4** | 0.140 | 0.033 | **0.775** |
| state correct (`r_action`) | 0.253 | 0.203 | 0.908 |
| reported correctly (`r_output`) | 0.563 | 0.446 | 0.870 |
| oracle progress | 0.253 | 0.203 | 0.911 |
| tool calls / episode | 1.12 | 1.33 | 3.51 |
| used a tool at all | 0.86 | 0.50 | 1.00 |
| illegal writes / episode | 0.050 | 0.002 | 0.171 |

Per task family, and by failure mode:

| Task family | Base | + SFT | + SFT + GRPO |
|---|---|---|---|
| `cancel_order` | 0.00 | 0.00 | 0.68 |
| `modify_address` | 0.51 | 0.21 | 1.00 |
| `return_items` | 0.00 | 0.00 | 0.93 |
| `lookup_and_report` | 0.32 | 0.00 | 0.72 |
| `refuse_invalid` | 0.00 | 0.00 | 0.89 |
| `exchange_items` *(never seen in RL)* | 0.01 | 0.01 | 0.55 |

| Failure mode | Base | + SFT | + SFT + GRPO |
|---|---|---|---|
| no tool call | 13.8% | 49.8% | 0.0% |
| stopped early | 67.2% | 38.2% | 6.3% |
| illegal write | 5.0% | 0.2% | 2.1% |
| acted, reported badly | 0.0% | 7.8% | 11.4% |
| **solved** | **14.0%** | **3.7%** | **79.4%** |

### Reading the numbers

**SFT made the model substantially worse — from 0.140 to 0.037.** This is the most
interesting result in the run, and the cause is specific rather than a training bug. The
SFT checkpoint stops calling tools: "used a tool at all" falls from 0.86 to 0.50, and half
of all episodes end with no tool call whatsoever. What it does instead is ask the user a
question:

> *"I cannot cancel an order or provide a refund without the user's user ID. Could you
> please provide me with your user ID so I can verify the order and proceed?"*

That behaviour is **correct in APIGen-MT and fatal here.** APIGen-MT trajectories are
collected against a user simulator that answers follow-up questions, so requesting missing
information is a rewarded move. `tau-retail-lite` has no simulator — the user's instruction
is a single turn, and the email needed for `find_user_id_by_email` is already in it. Asking
a question therefore ends the episode with nothing done. In 8 of 12 sampled no-tool
transcripts the model asks rather than acts; the base model never does this. This is a
train/serve distribution mismatch, not an optimisation failure, and it is exactly the kind
of thing that a loss curve cannot show and an aggregate score cannot explain.

SFT was not uniformly harmful, which supports that reading: illegal writes drop 25× (0.050 →
0.002), and a failure mode absent from the base model appears — "acted, reported badly", 7.8%
of episodes reaching the correct database state but failing to report it. The model learned
policy compliance and multi-step execution, then learned to hedge instead of starting.

**GRPO recovered the loss and went far past the base model, 0.037 → 0.794.** It repaired
precisely the failure the taxonomy identified: "no tool call" goes to 0.0% and "stopped
early" collapses from 38.2% to 6.3%, while tool calls per episode rise from 1.33 to 3.51 —
the model finally chains. This is the argument for online RL in one number: the environment
scores what actually happens, so a behaviour that is rewarded in the SFT corpus but useless
at deployment gets unlearned, which no amount of additional imitation on the same corpus
would achieve.

**The transfer result is the strongest evidence the gain is real.** `exchange_items` is
excluded from RL training entirely. It rises 0.01 → 0.55, well below the 0.84 average of the
five trained families but far above where it started. The gap is what genuine
generalisation looks like: the model learned a reusable procedure (identify the user, read
the order, act, report) rather than five memorised family-specific scripts. Combined with
disjoint train/test seeds, that makes memorisation an implausible explanation.

**One number moved the wrong way, and it matters.** Illegal writes per episode rose from
0.050 to 0.171 — RL made the model far more willing to act, and the violation penalty did
not fully offset that. The composite reward still improves overall, so GRPO is happy to buy
a large gain in completion at the price of some policy violation. In a real deployment
that trade is likely unacceptable, and it is a straightforward argument for a much heavier
violation penalty, or for treating violations as an episode-terminating constraint rather
than a subtracted term.

**Caveat on interpreting 0.794.** GRPO optimises the same grounded reward the evaluation
scores, so the RL checkpoint is directly trained on the eval metric in a way the base and
SFT checkpoints are not. The held-out family and disjoint seeds control for memorising
*tasks*, not for the objective and the metric coinciding. The honest claim is "RL taught the
model to do this environment's job well", not "RL made the model 5.7× better at tool use in
general" — §6 lists what would be needed to support the stronger claim.

---

## 4. Diagnosing the base model

An aggregate of 0.140 says the checkpoint is bad, not what to fix. Every episode is stored
and classified into one mutually-exclusive bucket by the first thing that went wrong
(`scripts/error_taxonomy.py`), which is what turned the headline into an actionable target.

The striking entry in the base column of §3 is what is *absent*. Malformed calls and
hallucinated tool names are **0%**, and no episode ever hit the turn limit: all 2,400 ended
because the model chose to answer. The base model's problem is not that it cannot spell a
tool call — it is that it does not take enough turns. Tool calls per episode: 332 episodes
made none, 1,442 made exactly one, 621 made two, and 5 made three. The behaviour is a single
lookup followed by an immediate answer, whether or not the task required a write.

Among the 1,792 early-stopping failures, `r_progress` is `0.0` in *every single one* — not
one correct oracle action with correct arguments. Outcomes are near-bimodal: an episode
either does the whole task or achieves nothing. That is the regime where GRPO's
group-relative advantage collapses, since identical scores across a group cancel, and it is
why the shaping terms in §2 rather than the outcome reward had to carry the early gradient.

**This diagnosis is what the RL result vindicates.** The taxonomy named premature
termination as the thing to fix; after GRPO, "no tool call" is 0.0%, "stopped early" falls
from 67.2% to 6.3%, and tool calls per episode rise from 1.12 to 3.51. The metric that moved
is the one the analysis pointed at, which is the strongest available evidence that the gain
came from the intended mechanism rather than luck.

One qualitative failure is worth recording because the taxonomy hides it inside "no tool
call": the model sometimes emits a correct call in the wrong wrapper.

```
[user]      Hi, I need to cancel an order. My email is yusuf.muller93@example.com ...
[assistant] <search> {"name": "find_user_id_by_email",
                      "arguments": {"email": "yusuf.muller93@example.com"}} </search>
```

The tool and the JSON are both right; only `<search>` instead of `<tool_call>` is wrong, so
nothing executes.

---

## 5. Engineering findings

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
- **A Modal spend limit is reported as "waiting for capacity."** Transient scarcity and a
  dead budget look identical in the logs, and the difference is fifteen minutes of waiting.

---

## 6. Honest limitations

- **The headline benchmark is one I wrote.** Train and test share no database and one task
  family is held out of RL entirely, which controls for memorisation but not for the
  environment being easier, or differently shaped, than a real benchmark.
- **BFCL was cut.** It was planned as the external, uncontaminated check and is genuinely
  suitable (fully offline, deterministic, no judge). It was dropped when the A10 fallback
  and the debugging above consumed the slack. Without it, nothing here demonstrates that
  gains transfer off-distribution.
- **RL is trained on the eval metric.** The most important caveat, restated: GRPO optimises
  the same grounded reward the harness scores. Disjoint seeds and a held-out family rule out
  memorising tasks, but not the objective and the metric being the same function. Base and
  SFT get no such advantage, so the 0.140 → 0.794 comparison is not between equals.
- **Single seed, one run per stage.** No error bars, and 2,400 episodes per checkpoint
  constrain sampling noise but say nothing about run-to-run variance in training.
- **The SFT stage is under-trained.** 63 steps over 1,000 trajectories, at batch size 1, is
  small. The regression in §3 is explained by a specific and verifiable behaviour rather than
  by undertraining, but more SFT might have changed its sign, and that was not tested.
- **The RL run is small.** 30 steps, ~480 tasks, ~3,840 rollouts is smoke-test scale for
  GRPO. That it worked this well at this size is itself surprising and deserves a repeat
  before being trusted.
- **A residual train/inference mismatch remains.** During a rollout the empty `<think>`
  block prefixes only the turn being generated and vanishes from re-rendered history. A
  single contiguous SFT sequence cannot reproduce that; training without think blocks keeps
  the gap to a constant 5-token prefix rather than spreading it across the history. It is
  asserted exactly in `tests/test_masking.py` so it cannot silently widen.

---

## 7. What a week would buy

Ordered by expected value, not by effort.

1. **Run the RL-only arm, which the results made the decisive experiment.** Since SFT
   *hurt*, "is SFT needed at all?" is now an open and cheap question rather than a
   completeness exercise. The arm is already implemented — `--adapter none` builds a fresh
   LoRA from the same `tooluse.train.lora` config SFT uses, so the arms differ in
   initialisation alone. It was launched and died with the compute budget. This is one run,
   and it would either justify the SFT stage or delete it.
2. **Fix the illegal-write regression.** RL tripled policy violations (0.050 → 0.171) while
   improving the composite reward, which means the current weighting sells compliance for
   completion. I would make violations episode-terminating rather than a subtracted term and
   re-measure both numbers, since a deployable agent cannot trade them off this way.
3. **Fix the SFT data mismatch rather than the SFT hyperparameters.** §3 shows the corpus
   teaches asking a user simulator that does not exist at deployment. Two clean options:
   filter APIGen-MT to trajectories that never request missing information, or add a scripted
   user to the environment so the behaviour becomes viable instead of fatal. The first is a
   day; the second is more faithful to τ-bench.
4. **Run BFCL multi-turn** on every checkpoint, plus real τ-bench with an LLM user
   simulator, to find out whether any of this transfers off-distribution. This is the check
   that would let the 0.794 be described as tool use rather than as this environment.
5. **Ablate the reward.** Every shaping term in §2 is a *claim* backed by construction and a
   unit test, not by measurement. Drop each in turn and watch both final success and whether
   the term induces hacking.
6. **Train for pass^k rather than pass^1.** Consistency is what makes a small agent
   deployable, and pass^k is already computed.
