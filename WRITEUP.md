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
   metric to watch for a longer run (§6).

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

**SFT made the model worse, 0.140 → 0.037, for a diagnosable reason.** The SFT checkpoint
stops calling tools — "used a tool at all" falls from 0.86 to 0.50 — and what it does
instead is ask a question:

> *"I cannot cancel an order or provide a refund without the user's user ID. Could you
> please provide me with your user ID so I can verify the order and proceed?"*

That is **correct in APIGen-MT and fatal here.** Those trajectories are collected against a
user simulator that answers follow-ups, so requesting missing information is rewarded.
`tau-retail-lite` has no simulator: the instruction is a single turn and already contains
the email `find_user_id_by_email` needs, so asking ends the episode having done nothing. In
8 of 12 sampled no-tool transcripts the model asks rather than acts; the base model never
does. This is a train/serve distribution mismatch, not an optimisation failure — the kind of
thing a loss curve cannot show and an aggregate cannot explain.

SFT was not uniformly harmful, which supports that reading: illegal writes fall 25×
(0.050 → 0.002) and a new bucket appears, "acted, reported badly" at 7.8% — correct database
state, no report. It learned policy compliance and multi-step execution, then learned to
hedge instead of starting.

**GRPO recovered the loss and went well past base, 0.037 → 0.794**, repairing exactly the
failure the taxonomy named: "no tool call" to 0.0%, "stopped early" from 38.2% to 6.3%, tool
calls per episode from 1.33 to 3.51. The model finally chains. This is the case for online
RL in one number — the environment scores what actually happens, so a behaviour rewarded by
the SFT corpus but useless at deployment gets unlearned, which more imitation on that same
corpus could not achieve.

**The transfer result is the best evidence the gain is real.** `exchange_items`, excluded
from RL entirely, rises 0.01 → 0.55 — well below the 0.84 average of trained families, far
above where it started. That gap is what generalisation looks like: a reusable procedure
(identify user, read order, act, report) rather than five memorised scripts. With disjoint
seeds, memorisation is not a plausible explanation.

**One number moved the wrong way.** Illegal writes rose 0.050 → 0.171. RL made the model far
more willing to act and the violation penalty did not offset it, so GRPO bought completion
at the price of compliance. For a real deployment that trade is unacceptable, and it argues
for violations terminating an episode rather than being subtracted from it.

**Caveat on 0.794.** GRPO optimises the same reward the harness scores, so the RL checkpoint
is trained on the eval metric in a way base and SFT are not. Disjoint seeds and a held-out
family control for memorising *tasks*, not for objective and metric coinciding. The honest
claim is that RL taught the model to do this environment's job well, not that it is 5.7×
better at tool use in general.

**How the target was chosen.** An aggregate of 0.140 says a checkpoint is bad, not what to
fix, so every episode is classified into one mutually-exclusive bucket by the first thing
that went wrong (`scripts/error_taxonomy.py`). The informative entry in the base column is
what is *absent*: malformed calls and hallucinated tools are **0%**, and no episode hit the
turn limit — all 2,400 ended because the model chose to answer. It is not that the model
cannot spell a tool call; it does not take enough turns. Among the 1,792 early stops,
`r_progress` is `0.0` in *every one*, so outcomes are near-bimodal — whole task or nothing,
the regime where group-relative advantage collapses. The taxonomy named premature
termination as the target, and that is precisely the metric RL moved, which is the best
available evidence the gain came from the intended mechanism rather than luck.

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
- **A Modal spend limit is reported as "waiting for capacity."** Transient scarcity and a
  dead budget look identical in the logs, and the difference is fifteen minutes of waiting.

---

## 5. Honest limitations

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

## 6. What a week would buy

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
