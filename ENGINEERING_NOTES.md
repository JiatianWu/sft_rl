# Engineering findings

Things that cost real time during the build. Each was found by running the producing code
rather than trusting a doc, and each is pinned by a test or a comment in the repo so it
cannot silently regress. Summarised in `WRITEUP.md` §5.

## TRL and the environment interface

**`environment_factory` probes the environment before using it.** TRL calls
`inspect.getmembers` on a freshly constructed instance, which evaluates every property. An
environment that raised `"reset() must be called first"` therefore crashed during *trainer
construction*, not at rollout time, which makes the traceback point at the wrong thing.
Environments must be valid straight out of `__init__`. Pinned by
`tests/test_env.py::test_fresh_instance_survives_trl_probe`.

**TRL's documented vLLM ceiling is stale.** The docs and the import check name 0.17.0–0.26.0
and warn on anything newer, but on TRL 1.10.0 an out-of-range vLLM only emits a
`UserWarning`. Verified on an A10: vLLM 0.27.1 generates, and GRPO trains with
`vllm_mode="colocate"` plus a LoRA adapter. `pyproject.toml` pins what was tested, not what
the docs claim.

**`GRPOConfig` moves between versions.** `max_prompt_length` no longer exists on 1.10.0;
`vllm_max_model_length` is the current knob. Likewise `TrainingArguments.warmup_ratio` is
gone in `transformers` 5.x, replaced by `warmup_steps`.

## Chat templates and the loss mask

**Qwen3's template has no `{% generation %}` marker**, so TRL's `assistant_only_loss` cannot
be used and masking is done by hand. This matters more than it sounds: training on
tool-response tokens teaches the model to *generate* database contents, which is exactly the
hallucination the environment exists to punish.

**The template renders the last assistant message differently from earlier ones.** It
injects an empty `<think>` block only on the final turn, so `render(messages[:i])` is *not*
a prefix of `render(messages[:j])`. Naive incremental diffing therefore misaligns the loss
mask silently — the training still runs, it just teaches the wrong tokens. Appending a
sentinel user turn before rendering removes the special case.

**Thinking had to be suppressed for a fair baseline.** With thinking enabled, Qwen3-0.6B
spends its entire completion budget reasoning and never emits a tool call. Left on, it would
have handicapped the *base* checkpoint specifically and inflated the apparent gain from SFT,
which is the one number the assignment most wants to be honest. `enable_thinking=False` is
used for all checkpoints, and the residual 5-token train/inference gap is asserted exactly
in `tests/test_masking.py` so it cannot widen unnoticed.

**The tools block dominates the SFT sequence.** APIGen-MT's policy text plus tool schemas is
~3,750 tokens. At a 4k cap only 6% of trajectories fit whole, and what got truncated was the
assistant turns — the only tokens carrying loss. At 8k, 97% fit. Even then only ~11% of
tokens contribute to the objective.

## Tests that cannot fail

**A fixture derived from the thing under test proves nothing.** `test_oracle_gets_full_reward`
is the most load-bearing test in the repo — its docstring says an unsolvable task makes every
RL number meaningless — and half of it was circular. The oracle's reply was built by joining
`required_outputs` together, then asserted to contain `required_outputs`. That checks the
substring matcher against itself and never against the environment, so it stayed green while
28 evaluation episodes were unwinnable.

The bug it hid: an order can list the same `item_id` on several lines (7.8% of generated
orders), `return_delivered_order_items` refunds every matching line, but the task required a
single unit price — so the agent reported the tool's own number, 130.0 against a required 65,
and was scored wrong. `test_required_facts_are_obtainable_from_the_tools` now scores the actual
tool return values across the full evaluation split, and fails on seed 100024 without the fix.

Worth noting the aggregate never showed it. What showed it was `return_items` reading exactly
0.93 in three independently trained arms — a number too stable across different checkpoints to
be a property of any of them.

## Running on a metered free tier

**H100/A100/L40S all require a payment method**, so everything ran on an **A10**. That image
needed two extra things: `CUDA_HOME` pointed at the pip-installed CUDA toolkit (the slim
image ships no system `nvcc`, which vLLM needs at startup), and the FlashInfer sampler
disabled — A10 is sm_86, which has no prebuilt FlashInfer kernels, so vLLM otherwise tries
to JIT-compile them on every cold start without a host C++ toolchain.

**SFT runs at batch size 1.** Qwen3's 151k vocabulary makes the logits tensor dominate
memory at 8192 tokens; batch 2 tried to allocate 8 GiB for cross-entropy alone. Effective
batch size is preserved with 16 gradient accumulation steps.

**A spend limit surfaces as a scheduling message, not a billing error.** Mid-SFT the job was
evicted at step 49/125, and the log said only that it was *"waiting to be scheduled on a
GPU_A10G worker … we are actively working on acquiring more capacity"*. That reads as
transient scarcity, so the correct response (stop waiting, change something) looks exactly
like the wrong one. Only an unrelated command surfaced the real cause: `Workspace ... has
exceeded its spend limit`.

**And the spend limit is not the credit balance.** The natural reading of that error is "the
credit is gone", which was wrong by more than an order of magnitude: `modal billing summary`
showed **$1.40 of ~$30 spent** at the point everything stopped. A free workspace carries a
separate spend cap that binds long before the credit does. Adding a payment method lifts the
cap and is the right fix either way, but diagnosing it as an exhausted balance would send you
looking for a new account instead of a settings change. Check `modal billing summary` before
believing any story about why a job will not schedule.

**The whole project cost $2.08**, of which $1.91 was A10 GPU time — and roughly $0.57 of that
was burned by the preempted SFT run and a stopped ablation. The four stages that produced the
reported numbers cost $0.89 together.

**Checkpoint per stage, not per pipeline.** The eviction cost the entire SFT run, because
the adapter is only written at the end. Worse, splitting the pipeline across four Modal
functions meant four cold starts, each re-downloading the base weights — roughly fifteen
minutes of paid GPU time spent on nothing. The stages now run in one container
(`modal_app.py::finish`) and commit the volume after each. On a metered budget this is the
difference between losing one stage and losing everything.
