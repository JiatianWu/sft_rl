# Plan: SFT → GRPO → Eval for a sub-4B tool-use agent

**Status:** plan, written before any code. **Budget:** 4 hours wall-clock, hard cap. **Compute:** Modal, $30 credit.

The goal is a *closed loop that is honestly measured*, not a high score. A 0.6B model doing multi-turn
tool use is going to be bad in absolute terms; the deliverable is a pipeline where each stage's
contribution is separable and the reward is grounded in verifiable state rather than an LLM judge.

---

## 1. Success criteria

I will consider this done if, at the 4-hour mark, the repo contains:

1. One command that runs SFT, one that runs GRPO, one that runs eval, all on Modal.
2. Three evaluated checkpoints — base, +SFT, +SFT+GRPO — measured with **identical** decoding and prompt
   settings, on a benchmark whose reward is computed from environment state.
3. A `results/` directory with raw per-task JSON, not just summary numbers.
4. A 2–3 page writeup that reports whatever actually happened, including regressions.

Explicit non-goals: beating any leaderboard, multi-GPU scaling, a novel algorithm.

---

## 2. Design decisions and trade-offs

| Choice | Decision | Why, and what I rejected |
|---|---|---|
| Base model | **Qwen3-0.6B** | Native `<tool_call>` chat template, so tool syntax is a formatting prior the base model already has — SFT teaches *when* to call, not *how to spell it*. Thinking mode is switchable off, which halves rollout length. Rejected Gemma E2B: heavier per rollout and a weaker tool-call template story. Qwen3-1.7B is the stretch target if time remains. |
| SFT method | **LoRA r=32 via Unsloth** | Assignment's suggested stack, ~2x faster and lower VRAM. Pre-committed fallback: TRL `SFTTrainer` + PEFT if the Unsloth image misbehaves (see gate T1). Unsloth is used **only** in the SFT container — I will not let it patch the GRPO process. |
| SFT data | **`Salesforce/APIGen-MT-5k`** | 5k *verified* multi-turn trajectories with real tool calls and tool responses, generated in the τ-bench **retail and airline** domains. That domain overlap with my RL environment is deliberate: SFT teaches the multi-turn tool protocol in a nearby domain without training on the RL tasks themselves. Fallbacks: `xlam-function-calling-60k` (single-turn, weaker) or Hermes function-calling (multi-turn but noisier). |
| RL algorithm | **GRPO** | Critic-free, so no value network to fit on a tiny model with a tiny budget — the single biggest wall-clock saving available. Group-relative baseline is well matched to a binary task reward. Rejected PPO (needs a critic) and DPO (offline, not "online RL against an environment"). |
| RL framework | **TRL `GRPOTrainer` with `environment_factory`** | Verified present in released TRL v1.9.0/v1.10.0: the trainer instantiates a stateful env per rollout, exposes its public methods as tools, calls `reset()` for the task, runs the tool loop, and calls `get_reward()` for the env-owned reward. This means I do not hand-write a multi-turn rollout loop or the tool-response token masking — the highest-risk code I would otherwise own. Rejected SkyRL: more capable at scale, more setup cost than 4 hours affords. |
| Environment | **`tau-retail-lite`, built in-repo** | A reimplementation of a τ-bench-retail-style tool surface (seeded DB of users/orders/products, ~10 tools) with **procedurally generated** tasks. Built rather than vendored because (a) real τ-bench requires an LLM user simulator — API cost, nondeterminism, and latency inside the RL loop, (b) procedural generation gives thousands of tasks with a clean train/test split, which a fixed 115-task benchmark cannot. |
| Primary eval | Held-out split of the same env | In-domain, high signal, cheap. |
| External eval | **BFCL v3 multi-turn** (`bfcl-eval`) | Fully offline, no LLM judge, no API key: state-based checking of the backend object after each turn plus response-based checking of the call trajectory. Zero contamination with my env. Rejected τ³-bench as primary for the user-simulator reason above. |

**The one trade-off I want to flag as load-bearing:** building my own environment makes the RL signal
clean and the train/test split honest, but it means my headline number is on a benchmark I designed.
That is a real weakness and I mitigate it by reporting BFCL alongside, where I control nothing.

---

## 3. Environment and reward design

This is where the assignment's weight is, so it gets specified before I write any training code.

### Tools and state

`RetailEnv` holds a seeded database (users, orders with line items, products, payment methods). Public
methods become tools: `find_user_by_email`, `get_user_details`, `get_order_details`,
`list_user_orders`, `get_product_details` (reads) and `cancel_pending_order`,
`modify_pending_order_address`, `return_delivered_order_items`, `exchange_delivered_order_items`,
`transfer_to_human` (writes). `reset(seed, task_family, ...)` builds the DB and returns the user
instruction as the prompt — the full request is given upfront, so no user simulator is needed.

### Task families

`cancel_order`, `return_items`, `exchange_items`, `modify_address`, `lookup_and_report` (read-only, must
report a computed number back to the user), and `refuse_invalid` (the requested write is not permitted
by policy — the agent must decline and escalate). The last family exists specifically so that "always
write something" is a losing strategy.

### Reward

The outcome term follows τ-bench's formulation, which is the right one: reward is the product of a
**state check** and an **output check**.

```
r_outcome  = r_action * r_output          # both in {0,1}
  r_action = (final DB state == ground-truth DB state)
  r_output = (all required strings appear in the agent's messages to the user)
```

Binary outcome alone is far too sparse for a 0.6B, so total reward is shaped:

| Component | Range | Purpose |
|---|---|---|
| `r_outcome` | {0, 1} | The grounded, verifiable objective. Dominant weight. |
| `r_progress` | [0, 1] | Fraction of the oracle write-actions performed **with correct arguments**. Partial credit that a wrong write cannot farm. |
| `r_format` | [0, 1] | Parseable tool call, existing tool name, arguments satisfying the schema. This is the 0.6B's dominant early failure mode and the densest available gradient. |
| `p_efficiency` | ≤ 0 | Penalty per redundant/duplicate call and per step beyond the oracle length. **Applied only when `r_outcome == 1`**, so it can never outrank correctness. |
| `p_violation` | ≤ 0 | Hallucinated tool name, or a write executed in a `refuse_invalid` task. |

Format weight is annealed (~0.3 → ~0.05) as validity saturates, using the `trainer_state` that TRL
passes to reward functions. `r_outcome`/`r_progress` come from the env's `get_reward`; the text-only
format checks are separate `reward_funcs`, since TRL sums env-owned and trainer-owned reward sources.

### Handling sparse signal — the specific failure mode

GRPO's advantage is group-relative, so if all G rollouts for a prompt get the same reward, the advantage
is zero and that prompt contributes **nothing**. At 0.6B, the default failure is "all G fail". Plan:

1. **Shaping above**, so groups differentiate on format and partial progress long before any full success.
2. **Difficulty-matched sampling** — start on 1–2 write-action tasks, introduce longer ones as success rises.
3. **Monitor `frac_reward_zero_std`**, which TRL logs. If it stays high, the run is burning compute on
   dead groups and I change the task mix rather than the learning rate.
4. **G = 8** to raise the odds of at least one success per group.
5. **SFT first is not optional.** Its real job is lifting the base success rate off the floor so GRPO
   has variance to exploit. If post-SFT env success is ~0%, GRPO cannot work and I fix SFT first.

### Anti-reward-hacking

State-hash checking defeats "describe the right answer without doing it"; the required-output check
defeats "mutate the DB silently"; success-gated efficiency penalty defeats tool spam; argument-level
validation defeats calling the right tool with garbage; the `refuse_invalid` family defeats a
write-happy policy.

---

## 4. Evaluation protocol

Three checkpoints — **base**, **+SFT**, **+SFT+GRPO** — under identical decoding, identical chat
template, `enable_thinking=False` everywhere (a mismatch here would silently invalidate the comparison).

- **A. Held-out env tasks** (primary, in-domain): 100 tasks, disjoint seeds *and* disjoint DB instances
  from training. 4 trials each → report **pass^1 … pass^4**, because reliability is the interesting
  property of a small agent, not best-of-N. Decomposed into `r_action`, `r_output`, tool-call validity,
  and mean steps, so I can say *which* part improved.
- **B. Compositional hold-out**: one task family excluded from RL training entirely. Does GRPO improve
  the protocol or just memorize the trained families?
- **C. BFCL v3 multi-turn** (external, OOD): `multi_turn_base` at minimum. Deterministic, no judge.
- **D. BFCL non-live AST** (regression): did SFT/RL damage single-turn calling?

I expect low absolute numbers on C and a real floor-effect risk. I would rather report a flat external
number honestly than tune until it moves.

---

## 5. Execution timeline with kill-gates

Each gate has a pre-committed fallback so I never debug past the clock.

| Time | Work | Gate / fallback |
|---|---|---|
| 0:00–0:30 | Repo scaffold, two Modal images, smoke-test Qwen3-0.6B generation. **Verify §7 assumptions against installed code.** | **T1:** image build > 10 min → drop Unsloth, use TRL SFTTrainer. |
| 0:30–1:15 | Env, task generator, verifier, eval harness. Run **base** eval (A). | **T2:** env not done by 1:15 → cut `exchange_items` and `modify_address`, ship 4 families. |
| 1:15–1:50 | APIGen-MT → chat-template conversion with assistant-only loss; LoRA SFT; eval **+SFT**. | **T3:** dataset gated/unavailable → xLAM-60k. **T4:** post-SFT env success ≈ 0 → fix SFT before touching GRPO. |
| 1:50–3:00 | GRPO on env, adapter initialized from the SFT adapter. Eval **+SFT+GRPO**. | **T5:** colocate+PEFT hangs (a known TRL issue on some GPUs) → `vllm_mode="server"`, then transformers continuous batching. **T6:** reward collapses → drop to format+progress only and report it. |
| 3:00–3:30 | BFCL (C, D) on all three checkpoints. | **T7:** harness fights back → report A and B only, document why. |
| 3:30–4:00 | Writeup, plots, repo polish. | Non-negotiable; the writeup ships even if RL failed. |

A failed GRPO run that is *diagnosed* is a better artifact than a suspiciously good number, and the
writeup is structured to allow that outcome.

---

## 6. Compute budget

Modal, per-second billing, $30 starter credit. H100 $3.95/hr, A100-80GB $2.50, L40S $1.95, A10 $1.10.

| Job | GPU | Est. time | Est. cost |
|---|---|---|---|
| Smoke tests | A10 | 0.2 h | $0.25 |
| SFT (LoRA, 0.6B) | H100 | 0.3 h | $1.20 |
| GRPO | H100 | 1.2 h | $4.75 |
| Evals (3 ckpts × 4 suites) | A100-80 | 1.0 h | $2.50 |
| Debugging / reruns headroom | — | — | ~$8 |
| **Total** | | | **~$17, ceiling $30** |

Comfortable margin. If GRPO needs more steps than planned, headroom goes there. Everything is a single
GPU — no multi-node coordination cost, in either dollars or minutes.

---

## 7. Assumptions to verify before writing code

Per my own rule about trusting the producer over the docs, separating what I actually checked from what
I only read.

**Verified by reading source / PyPI just now:**
- `environment_factory`, `reset`, `get_reward` exist in released TRL **v1.9.0 and v1.10.0** (read `grpo_trainer.py` at both tags).
- It is marked **experimental** — emits a warning unless `TRL_EXPERIMENTAL_SILENCE=1` — and requires `transformers>=5.2.0`. I am building on an API that may change.
- `max_tool_calling_iterations` exists as the turn-cap knob; `num_generations` defaults to 8.
- Latest: trl 1.10.0, transformers 5.15.1, vllm 0.27.1.

**Doc-only, must confirm at setup (gate T1):**
- TRL docs claim vLLM support spans 0.17.0–0.26.0 while PyPI latest is 0.27.1 → **pin vllm 0.26.x** and confirm.
- The colocate + PEFT hang reported on A100-class GPUs may already be fixed; I will test early, not at 2:30.
- APIGen-MT-5k is gated behind accepting terms; its exact schema is unconfirmed. Needs an HF token (none set locally).
- BFCL CLI flags (`--enable-lora`, `--lora-modules`) and its exact multi-turn scoring semantics.

---

## 8. Repo layout

```
src/tooluse/
  env/      retail.py  tasks.py  verify.py     # env, procedural tasks, reward/verifier
  data/     prepare_sft.py                     # APIGen-MT → chat template
  train/    sft_unsloth.py  grpo.py
  eval/     run_eval.py  bfcl_adapter.py
modal/      app.py                             # images + entrypoints
results/                                       # raw per-task JSON, committed
PLAN.md  FINDINGS.md  README.md
```

---

## 9. With a week instead of an afternoon

1. **Scale the base** to Qwen3-1.7B/4B and run the SFT/RL ablation at two sizes — does RL help more or
   less as the base gets stronger?
2. **Real τ³-bench**, LLM user simulator included, as the headline eval, with pass^k over ≥4 trials.
3. **Ablate the reward**, one component at a time. Right now I am asserting the shaping terms help; with
   a week I would measure it, and measure whether shaping induces hacking.
4. **Separate SFT-only, RL-only, and SFT→RL** to show what RL contributes over more imitation, plus
   3 seeds for error bars. Single-seed deltas at this scale are close to meaningless.
5. **Async/off-policy GRPO** (or DAPO/GSPO) to stop wasting GPU on rollout-then-train serialization.
6. **Grow the env** toward multi-domain with TRL's environment routing, and add an LLM-generated task
   pipeline with automatic verifiability filtering (the APIGen-MT idea, applied to my own env).
7. **Error taxonomy** over failed trajectories — wrong tool, right tool/wrong args, premature stop,
   policy violation — because the aggregate pass rate says nothing about what to fix next.
8. **Train for pass^k directly** rather than pass^1; consistency is what makes a small agent deployable.
9. **Contamination audit** between SFT data and every eval set.

---

## 10. Open questions I'd raise with the team

- Is a self-built environment acceptable as the primary benchmark given the user-simulator cost, or would
  you rather see lower-confidence numbers on real τ³-bench?
- Is the 4-hour cap wall-clock including GPU wait, or hands-on time?
