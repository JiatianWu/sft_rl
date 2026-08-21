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

**`Qwen3-0.6B`.** Its chat template already emits `<tool_call>` blocks, so tool syntax is a
prior the base model has some grip on and SFT can spend its capacity on *when* to call
rather than *how to spell it*. It is also small enough that a GRPO step with 8 rollouts is
seconds, not minutes, which is what makes an RL run fit in the budget at all.

**GRPO over PPO.** GRPO is critic-free. Fitting a value head on a 0.6B model with a few
hundred episodes would be the least reliable part of the system, and skipping it is the
single biggest saving available. The group-relative baseline also matches a binary task
reward well.

**A self-built environment over τ-bench itself.** τ-bench's reward design is the right one
and is copied directly (below), but running the benchmark requires an LLM user simulator:
API cost, nondeterminism, and latency inside the RL loop. A procedural environment gives
thousands of tasks, a train/test split with no shared database, and a free, deterministic
reward. The cost is real and stated plainly in §6: the headline number is measured on a
benchmark I wrote.

**Stock PEFT instead of Unsloth.** Unsloth was the suggested stack and would have made SFT
roughly twice as fast. It pins its own `transformers`, while TRL's `environment_factory`
requires `transformers>=5.2`. One dependency set removed a version conflict and an image
build from a hard time budget; the ~10 minutes saved did not justify the risk.

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

Aggregate scores do not explain failures, so the harness stores full transcripts. The base
checkpoint's dominant failure is not reasoning — it is protocol:

```
[user]      Hi, I need to cancel an order. My email is yusuf.muller93@example.com ...
[assistant] <search> {"name": "find_user_id_by_email",
                      "arguments": {"email": "yusuf.muller93@example.com"}} </search>
```

The tool is right and the JSON is well-formed, but it is wrapped in `<search>` rather than
`<tool_call>`, so nothing executes and the episode ends. Across 480 base episodes, 76 made
no tool call at all, 288 made exactly one, and only 4 reached three. The base model does not
chain: it calls one tool, then answers without using the result.

This is why the protocol term carries real weight early in RL, and why SFT — which shows the
model thousands of correctly-wrapped multi-turn trajectories — is the stage expected to move
this particular number.

---

## 5. Engineering findings worth recording

These cost time, and each was found by running the producer rather than trusting a doc.

- **TRL's `environment_factory` probes the environment before using it.** It calls
  `inspect.getmembers` on a freshly constructed instance, which evaluates properties. An
  environment that raised "call reset() first" therefore crashed during *trainer
  construction*. Environments must be valid from `__init__`.
- **Qwen3's chat template has no `{% generation %}` marker**, so TRL's `assistant_only_loss`
  cannot be used with it. Masking is done manually. This matters more than it sounds:
  training on tool-response tokens teaches the model to *generate* database contents, which
  is precisely the hallucination the environment exists to punish.
- **The template renders the last assistant message differently from earlier ones** (it
  injects an empty `<think>` block). That makes `render(messages[:i])` *not* a prefix of
  `render(messages[:j])`, so naive incremental diffing silently misaligns the loss mask.
  Appending a sentinel user turn before rendering removes the special case.
- **Thinking had to be suppressed for a fair baseline.** With thinking enabled, Qwen3-0.6B
  spends its entire completion budget reasoning and never emits a tool call. Left on, it
  would have handicapped the *base* checkpoint specifically and inflated the apparent gain
  from SFT. `enable_thinking=False` is used for all three checkpoints.
- **The tools block dominates the SFT sequence.** APIGen-MT's policy plus tool schemas is
  ~3,750 tokens. At a 4k cap only 6% of trajectories fit whole and the assistant turns were
  being truncated away; at 8k, 97% fit. Only ~11% of tokens carry loss.
- **TRL's documented vLLM ceiling is stale.** It names 0.17.0–0.26.0 and warns on anything
  newer, but 0.27.1 only produces a `UserWarning`, and GRPO trained fine with colocate vLLM
  plus a LoRA adapter. Pins in `pyproject.toml` reflect what was tested, not what the docs
  claim.
- **Free-tier reality.** H100/A100/L40S all require a payment method, so everything ran on
  an **A10**. That image also needed `CUDA_HOME` pointed at the pip CUDA toolkit and the
  FlashInfer sampler disabled — A10 is sm_86, which has no prebuilt FlashInfer kernels, so
  vLLM otherwise tries to JIT-compile them on every cold start without a host toolchain.
  SFT runs at batch size 1 because Qwen3's 151k vocab makes the logits tensor dominate
  memory at 8192 tokens.
- **A spend limit surfaces as a scheduling message, not a billing error.** When the credit
  ran out mid-SFT, the job was evicted at step 49/125 and the log said only that it was
  *"waiting to be scheduled on a GPU_A10G worker … acquiring more capacity"*. That reads as
  transient scarcity, so the correct response (stop waiting) looks exactly like the wrong
  one. Only an unrelated command surfaced the actual cause, `exceeded its spend limit`.
- **Checkpoint per stage, not per pipeline.** The eviction cost the whole SFT run because
  the adapter is written once at the end. Worse, splitting the pipeline across four Modal
  functions meant four cold starts, each re-downloading the base weights — roughly fifteen
  minutes of paid GPU time spent on nothing. The stages now run in one container
  (`modal_app.py::finish`) and commit the volume after each, so an interruption costs one
  stage instead of the run. On a metered budget, this is the difference between losing
  twenty minutes and losing everything.

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

1. **Attack `frac_reward_zero_std` directly.** Half of all groups produced no gradient.
   Filtering tasks whose G rollouts all agree, and sampling difficulty against a running
   per-family success estimate, is the cheapest large win available.
2. **Run BFCL multi-turn** on all three checkpoints, plus real τ³-bench with an LLM user
   simulator, to find out whether any of this transfers.
3. **Ablate the reward.** Every shaping term above is currently a *claim* backed by
   construction and a unit test, not by measurement. I would drop each in turn and check
   both final success and whether shaping induces hacking.
4. **Separate SFT-only, RL-only and SFT→RL**, at 0.6B and 1.7B, with 3 seeds. This answers
   "what did RL add over more imitation?", which the current single path cannot.
5. **Scale the base model.** 0.6B is at the edge of being able to follow a tool protocol at
   all; a good deal of the measured variance is protocol noise rather than decision quality.
6. **Train for pass^k rather than pass^1.** Consistency is what makes a small agent
   deployable, and pass^k is already computed.
7. **Grow the environment** toward multiple domains using TRL's environment routing, and
   build an error taxonomy (wrong tool / right tool wrong args / premature stop / policy
   violation) so that a failing aggregate points at something actionable.
