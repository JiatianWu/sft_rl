# Multi-turn tool use on a 0.6B model: SFT → GRPO → eval

**Time budget:** 4 hours, hard cap. **Compute:** Modal, ~$30 credit, single GPU.
**Model:** `Qwen3-0.6B`. **Result artifacts:** `results/`, one JSON record per episode.

The aim was a loop that closes cleanly and is honestly measured, not a high score. A 0.6B
model doing multi-turn tool use is weak in absolute terms; what is worth building in four
hours is a pipeline where each stage's contribution is separable and the reward is grounded
in verifiable state rather than an LLM judge.

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
2. **A held-out easy difficulty**: every task states the order id, keeping the oracle to
   1–3 calls.
3. **SFT first is load-bearing, not decorative.** Its real job is lifting the success rate
   off the floor so RL has variance to exploit.
4. **G = 8**, to raise the chance of at least one success per group.
5. **`frac_reward_zero_std`**, which TRL logs, measures exactly this failure. In the GRPO
   run it sat around 0.5 — half of all groups produced no learning signal at all. That is
   the single clearest lever for a longer run (§7).

---

## 3. Results

All three checkpoints were evaluated with identical decoding (T=0.7, top-p 0.95, seed
fixed), identical prompts, and identical chat-template settings, on **120 held-out tasks ×
4 trials = 480 episodes**. Test seeds are drawn from a disjoint range, so no evaluation task
shares a database with any training task.

<!-- RESULTS_TABLE -->

### Reading the numbers

<!-- RESULTS_NARRATIVE -->

---

## 4. What the base model actually does

An aggregate of 0.10 says the checkpoint is bad, not what to fix, so every episode is stored
and classified into one mutually-exclusive bucket by the first thing that went wrong
(`scripts/error_taxonomy.py`):

| Failure mode | Base |
|---|---|
| stopped early — answered with oracle actions outstanding | 68.3% |
| no tool call — answered from the prompt alone | 15.8% |
| illegal write — a write the policy or order state forbids | 5.8% |
| **solved** | **10.0%** |

The striking entry is what is *absent*. Malformed calls and hallucinated tool names are
**0%**, and no episode ever hit the turn limit: all 480 ended because the model chose to
answer. The base model's problem is not that it cannot spell a tool call. It is that it
does not take enough turns.

The distribution of tool calls per episode makes this concrete — 76 episodes made none, 288
made exactly one, 112 made two, and 4 made three. The behaviour is: perform one lookup, then
immediately answer the user, whether or not the task required a write.

One qualitative failure is worth recording because the taxonomy hides it inside "no tool
call". The model sometimes produces a correct call in the wrong wrapper:

```
[user]      Hi, I need to cancel an order. My email is yusuf.muller93@example.com ...
[assistant] <search> {"name": "find_user_id_by_email",
                      "arguments": {"email": "yusuf.muller93@example.com"}} </search>
```

The tool and the JSON are both right; only `<search>` instead of `<tool_call>` is wrong, so
nothing executes.

**The consequence for RL is the important part.** Among the 368 early-stopping failures,
`r_progress` is `0.0` in *every single one* — not one correct oracle action with correct
arguments. Outcomes are therefore near-bimodal: an episode either does the whole task or
achieves literally nothing. That is precisely the regime where GRPO's group-relative
advantage collapses, because all G rollouts in a group score identically and cancel. It
predicts the `frac_reward_zero_std ≈ 0.5` observed during RL, and it is the concrete reason
SFT is load-bearing rather than decorative: its job is to manufacture the partial successes
that give RL something to differentiate.

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
- **Single seed, one run per stage.** No error bars. At this scale the differences between
  checkpoints should be read as directional, not precise.
- **The compute budget bound the results, not the code.** The $30 credit was exhausted
  partway through the SFT run (§5). Every stage had already been validated end to end on
  Modal beforehand — the GRPO smoke test confirms `environment_factory` trains with
  colocate vLLM and a LoRA adapter — so what is missing is GPU minutes, not working
  pipeline. Any number below that is absent rather than estimated is marked as such; none
  are projected.
- **The RL run is small.** A few hundred tasks and a few thousand rollouts is a smoke-test
  scale for GRPO, chosen to fit the budget on a slower GPU than planned.
- **A residual train/inference mismatch remains.** During a rollout the empty `<think>`
  block prefixes only the turn being generated and vanishes from re-rendered history. A
  single contiguous SFT sequence cannot reproduce that; training without think blocks keeps
  the gap to a constant 5-token prefix rather than spreading it across the history. It is
  asserted exactly in `tests/test_masking.py` so it cannot silently widen.

---

## 7. What a week would buy

Ordered by expected value, not by effort.

1. **Attack premature termination, the failure the data actually names.** §4 shows the
   problem is not syntax (0% malformed) but stopping after one call. The lever is the
   decision to continue, so I would reward reaching the *next* required action rather than
   only the final state, and check whether the progress term already does this or is too
   coarse at 1–3 oracle actions.
2. **Attack `frac_reward_zero_std`.** Half of all groups produced no gradient — the direct
   consequence of the bimodal outcomes in §4. Filtering groups whose rollouts all agree, and
   sampling difficulty against a running per-family success estimate, is the cheapest large
   win available.
3. **Finish the ablation grid: SFT-only, RL-only, SFT→RL**, at 0.6B and 1.7B, 3 seeds. This
   answers "what did RL add over more imitation?", which a single path cannot. The RL-only
   arm is already implemented (`--adapter none` builds a fresh LoRA from the same
   `tooluse.train.lora` config SFT uses, so the arms differ in initialisation alone) and was
   launched before the budget ran out: a run away, not a build away.
4. **Run BFCL multi-turn** on every checkpoint, plus real τ-bench with an LLM user
   simulator, to find out whether any of this transfers off-distribution.
5. **Ablate the reward.** Every shaping term in §2 is a *claim* backed by construction and a
   unit test, not by measurement. Drop each in turn and watch both final success and whether
   the term induces hacking.
6. **Train for pass^k rather than pass^1.** Consistency is what makes a small agent
   deployable, and pass^k is already computed.
