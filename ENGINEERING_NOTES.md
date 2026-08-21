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

## Running on a metered free tier

**H100/A100/L40S all require a payment method**, so everything ran on an **A10**. That image
needed two extra things: `CUDA_HOME` pointed at the pip-installed CUDA toolkit (the slim
image ships no system `nvcc`, which vLLM needs at startup), and the FlashInfer sampler
disabled — A10 is sm_86, which has no prebuilt FlashInfer kernels, so vLLM otherwise tries
to JIT-compile them on every cold start without a host C++ toolchain.

**SFT runs at batch size 1.** Qwen3's 151k vocabulary makes the logits tensor dominate
memory at 8192 tokens; batch 2 tried to allocate 8 GiB for cross-entropy alone. Effective
batch size is preserved with 16 gradient accumulation steps.

**A spend limit surfaces as a scheduling message, not a billing error.** When the credit ran
out mid-SFT, the job was evicted at step 49/125 and the log said only that it was *"waiting
to be scheduled on a GPU_A10G worker … we are actively working on acquiring more capacity"*.
That reads as transient scarcity, so the correct response (stop waiting, change something)
looks exactly like the wrong one. Only an unrelated command surfaced the actual cause:
`Workspace ... has exceeded its spend limit`. Worth knowing before you spend fifteen minutes
waiting politely for capacity that will never arrive.

**Checkpoint per stage, not per pipeline.** The eviction cost the entire SFT run, because
the adapter is only written at the end. Worse, splitting the pipeline across four Modal
functions meant four cold starts, each re-downloading the base weights — roughly fifteen
minutes of paid GPU time spent on nothing. The stages now run in one container
(`modal_app.py::finish`) and commit the volume after each. On a metered budget this is the
difference between losing one stage and losing everything.
