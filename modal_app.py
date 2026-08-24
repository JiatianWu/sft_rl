"""Modal entrypoints for the SFT -> GRPO -> eval pipeline.

One image for all stages. Keeping SFT and RL on the same dependency set avoids a
version conflict between Unsloth's pinned `transformers` and the `transformers>=5.2`
that TRL's `environment_factory` requires.

    modal run modal_app.py::prepare
    modal run modal_app.py::smoke
    modal run modal_app.py::sft
    modal run modal_app.py::evaluate --tag base
    modal run modal_app.py::grpo
"""

from __future__ import annotations

import os

import modal

APP_NAME = "tooluse-sft-rl"

# Overridable so the pipeline can fall back to whatever tier the account can reach
# (H100 requires a payment method on file; A10/L4/T4 do not).
GPU = os.environ.get("TOOLUSE_GPU", "A10")

CUDA_HOME = "/usr/local/lib/python3.12/site-packages/nvidia/cu13"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.15.1",
        "trl==1.10.0",
        "vllm==0.27.1",
        "peft==0.20.0",
        "datasets==5.0.1",
        "accelerate==1.14.0",
    )
    .env(
        {
            "HF_HOME": "/cache/hf",
            "TRL_EXPERIMENTAL_SILENCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            # vLLM JIT-compiles kernels at startup and needs nvcc. The slim image has no
            # system CUDA toolkit, but the pip `nvidia-cuda-nvcc` wheel ships one here.
            "CUDA_HOME": CUDA_HOME,
            "PATH": f"{CUDA_HOME}/bin:/usr/local/bin:/usr/bin:/bin",
            # A10 is sm_86, for which FlashInfer ships no prebuilt kernels, so vLLM tries
            # to JIT-compile them on every cold start. Using the native sampler skips a
            # compile that needs a full host toolchain and burns GPU minutes each launch.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_python_source("tooluse")
)

# BFCL gets its own image on purpose. `bfcl-eval` pulls its own transformers/vllm, and the
# training image above is a combination that took real work to get running on an A10. The worst
# outcome of sharing one image is no BFCL numbers *and* a broken pipeline.
bfcl_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "bfcl-eval[oss-eval-vllm]",
        # The extra pins `vllm==0.8.5` but leaves `transformers` free, so pip resolves 5.15.1
        # and the server dies at startup: vllm 0.8.5 reads `all_special_tokens_extended`, which
        # transformers v5 removed. 4.51.3 is the last-4.x line that still knows Qwen3.
        "transformers==4.51.3",
        # qwen_agent (imported by BFCL's Qwen handler) needs this and does not declare it;
        # without it `bfcl models` dies on `ModuleNotFoundError: No module named 'soundfile'`.
        "soundfile",
        "peft==0.20.0",
    )
    .env(
        {
            "HF_HOME": "/cache/hf",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
)

app = modal.App(APP_NAME, image=image)

cache = modal.Volume.from_name("tooluse-cache", create_if_missing=True)
workspace = modal.Volume.from_name("tooluse-workspace", create_if_missing=True)
VOLUMES = {"/cache": cache, "/work": workspace}

HOURS = 60 * 60


@app.function(volumes=VOLUMES, timeout=1 * HOURS)
def prepare(apigen: int = 0, hermes: int = 0, out: str = "/work/data/sft_apigen.jsonl") -> None:
    """Download and convert the SFT dataset into the shared volume.

    `hermes > 0` mixes in NousResearch/hermes-function-calling-v1, whose assistant turns make
    several calls at once — the one thing APIGen-MT and `tau-retail-lite` never demonstrate, and
    the reason BFCL scored the SFT arms 0/200 on parallel calling (WRITEUP.md §3.6).
    """
    from pathlib import Path

    from tooluse.data.prepare_sft import build

    build(
        limit=apigen or None,
        out_path=Path(out),
        cache=Path("/cache/apigen-mt_5k.json"),
        hermes=hermes,
    )
    workspace.commit()


@app.function(timeout=10 * 60)
def find_nvcc() -> None:
    """Locate the pip-installed CUDA toolkit so CUDA_HOME can point at it."""
    import subprocess

    print(subprocess.run(["find", "/usr/local/lib/python3.12/site-packages/nvidia", "-name", "nvcc"],
                         capture_output=True, text=True).stdout)
    print(subprocess.run(["find", "/usr/local/lib/python3.12/site-packages/nvidia", "-maxdepth", "2",
                          "-name", "include", "-o", "-maxdepth", "2", "-name", "bin"],
                         capture_output=True, text=True).stdout)


@app.function(gpu=GPU, timeout=10 * 60)
def probe() -> str:
    """Cheapest possible check that this account can actually schedule `GPU`."""
    import torch

    return torch.cuda.get_device_name(0)


@app.function(gpu=GPU, volumes=VOLUMES, timeout=1 * HOURS)
def smoke() -> None:
    """Fail fast on the assumptions that would otherwise break at 2:30 into the budget."""
    import torch
    import transformers
    import trl
    import vllm

    print(f"torch {torch.__version__} | transformers {transformers.__version__}")
    print(f"trl {trl.__version__} | vllm {vllm.__version__}")
    print(f"gpu: {torch.cuda.get_device_name(0)}")

    # 1. Does GRPOTrainer really accept environment_factory in this version?
    import inspect

    from trl import GRPOTrainer

    assert "environment_factory" in inspect.signature(GRPOTrainer.__init__).parameters
    print("environment_factory: present")

    # 2. Do the environment's tools render as schemas, and does a rollout score?
    from tooluse.env import RetailEnv
    from tooluse.eval.harness import tool_schemas

    env = RetailEnv()
    prompt = env.reset(seed=0, family="cancel_order", difficulty="easy")
    print(f"tools: {len(tool_schemas(env))} | prompt: {prompt[:90]}...")

    # 3. Can vLLM serve the base model and produce a tool call at all?
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from tooluse.data.masking import CHAT_TEMPLATE_KWARGS
    from tooluse.env.retail import SYSTEM_PROMPT

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    llm = LLM(model="Qwen/Qwen3-0.6B", max_model_len=16384, gpu_memory_utilization=0.6)
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        tools=tool_schemas(env),
        tokenize=False,
        add_generation_prompt=True,
        **CHAT_TEMPLATE_KWARGS,
    )
    out = llm.generate([text], SamplingParams(temperature=0.0, max_tokens=200), use_tqdm=False)
    print("--- base model first turn ---")
    print(out[0].outputs[0].text)
    cache.commit()


@app.function(gpu=GPU, volumes=VOLUMES, timeout=3 * HOURS)
def sft(limit: int = 2500, epochs: float = 1.0, max_length: int = 8192) -> None:
    import subprocess

    subprocess.run(
        [
            "python",
            "-m",
            "tooluse.train.sft",
            "--data",
            "/work/data/sft_apigen.jsonl",
            "--output",
            "/work/checkpoints/sft",
            "--limit",
            str(limit),
            "--epochs",
            str(epochs),
            "--max-length",
            str(max_length),
        ],
        check=True,
    )
    workspace.commit()


@app.function(gpu=GPU, volumes=VOLUMES, timeout=4 * HOURS)
def grpo(
    steps: int = 60,
    adapter: str = "/work/checkpoints/sft",
    protocol_weight: float = 0.3,
    output: str = "/work/checkpoints/grpo",
    grad_accum: int = 2,
) -> None:
    import subprocess

    command = [
        "python",
        "-m",
        "tooluse.train.grpo",
        "--output",
        output,
        "--grad-accum",
        str(grad_accum),
        "--steps",
        str(steps),
        "--protocol-weight",
        str(protocol_weight),
    ]
    if adapter and adapter != "none":
        command += ["--adapter", adapter]
    subprocess.run(command, check=True)
    workspace.commit()


@app.function(gpu=GPU, volumes=VOLUMES, timeout=2 * HOURS)
def evaluate(tag: str, adapter: str | None = None, n_seeds: int = 100, trials: int = 4) -> None:
    import subprocess

    command = [
        "python",
        "-m",
        "tooluse.eval.run_eval",
        "--tag",
        tag,
        "--n-seeds",
        str(n_seeds),
        "--trials",
        str(trials),
        "--out",
        "/work/results",
    ]
    if adapter and adapter != "none":
        command += ["--adapter", adapter]
    subprocess.run(command, check=True)
    workspace.commit()


@app.function(gpu=GPU, volumes=VOLUMES, timeout=4 * HOURS)
def finish(sft_limit: int = 1000, grpo_steps: int = 30, n_seeds: int = 100, trials: int = 4) -> None:
    """SFT -> eval -> GRPO -> eval inside a single container.

    Running these as four separate Modal functions costs four cold starts, and each one
    re-downloads the base weights: roughly fifteen minutes of paid GPU time spent on
    nothing. Sharing one container also shares the HF cache.

    The volume is committed after every stage. An earlier run was preempted mid-SFT and
    lost the lot, because the adapter is only written at the end; committing per stage means
    an interruption costs one stage rather than all of them.
    """
    import subprocess

    def stage(name: str, command: list[str]) -> None:
        print(f"\n{'=' * 70}\n[finish] {name}\n{'=' * 70}", flush=True)
        subprocess.run(command, check=True)
        workspace.commit()
        print(f"[finish] {name} committed", flush=True)

    stage(
        "1/4 SFT",
        [
            "python", "-m", "tooluse.train.sft",
            "--data", "/work/data/sft_apigen.jsonl",
            "--output", "/work/checkpoints/sft",
            "--limit", str(sft_limit),
            "--max-length", "8192",
        ],
    )
    stage(
        "2/4 eval post-SFT",
        [
            "python", "-m", "tooluse.eval.run_eval",
            "--tag", "sft", "--adapter", "/work/checkpoints/sft",
            "--n-seeds", str(n_seeds), "--trials", str(trials), "--out", "/work/results",
        ],
    )
    stage(
        "3/4 GRPO",
        [
            "python", "-m", "tooluse.train.grpo",
            "--adapter", "/work/checkpoints/sft",
            "--output", "/work/checkpoints/grpo",
            "--steps", str(grpo_steps),
        ],
    )
    stage(
        "4/4 eval post-RL",
        [
            "python", "-m", "tooluse.eval.run_eval",
            "--tag", "grpo", "--adapter", "/work/checkpoints/grpo",
            "--n-seeds", str(n_seeds), "--trials", str(trials), "--out", "/work/results",
        ],
    )
    print("\n[finish] all four stages complete", flush=True)


@app.function(gpu=GPU, volumes=VOLUMES, timeout=4 * HOURS)
def rebaseline(n_seeds: int = 100, trials: int = 4) -> None:
    """Re-evaluate every checkpoint on the current metric, in one container.

    Two scoring bugs were fixed after the original sweep: `return_items` had 7 unsatisfiable
    seeds, and order ids were only accepted with a leading `#`. Fixing them shifts every arm
    by roughly +0.01, which changes no conclusion — but leaving five arms on the old metric
    and one on the new means the headline table silently compares two different rulers.

    Re-running all six also produces failure transcripts for every arm under the corrected
    sampler. The originals sampled `results[:12]`, which turned out to be all passes, so the
    qualitative claims about *why* base and SFT fail rest on weaker evidence than the rest.

    One container: the base weights download once and the HF cache is shared, which is most of
    the wall clock. Results are committed after each arm so an interruption costs one arm.
    """
    import subprocess

    arms = [
        ("base", None),
        ("sft", "/work/checkpoints/sft"),
        ("rl_only", "/work/checkpoints/rl_only"),
        ("grpo", "/work/checkpoints/grpo"),
        ("rl_only_long", "/work/checkpoints/rl_only_long"),
        ("grpo_long", "/work/checkpoints/grpo_long"),
    ]
    for index, (tag, adapter) in enumerate(arms, start=1):
        print(f"\n{'=' * 70}\n[rebaseline] {index}/{len(arms)}: {tag}\n{'=' * 70}", flush=True)
        command = [
            "python", "-m", "tooluse.eval.run_eval",
            "--tag", tag,
            "--n-seeds", str(n_seeds),
            "--trials", str(trials),
            "--out", "/work/results",
        ]
        if adapter:
            command += ["--adapter", adapter]
        subprocess.run(command, check=True)
        workspace.commit()
        print(f"[rebaseline] {tag} committed", flush=True)
    print("\n[rebaseline] all six arms re-evaluated on one metric", flush=True)


@app.function(gpu=GPU, volumes=VOLUMES, timeout=5 * HOURS)
def rl_arm(
    tag: str,
    adapter: str = "none",
    steps: int = 30,
    n_seeds: int = 100,
    trials: int = 4,
) -> None:
    """One RL arm end to end: GRPO then evaluation, in a single container.

    Used for the two experiments the results made worth running — an RL-only arm
    (`adapter="none"`, a fresh LoRA) to find out whether the SFT stage earns its place given
    that it *lowered* success, and a longer run from the SFT adapter to test whether the
    residual failures are undertraining rather than design.
    """
    import subprocess

    output = f"/work/checkpoints/{tag}"
    train = [
        "python", "-m", "tooluse.train.grpo",
        "--output", output,
        "--steps", str(steps),
    ]
    if adapter and adapter != "none":
        train += ["--adapter", adapter]

    print(f"\n{'=' * 70}\n[{tag}] GRPO, {steps} steps, adapter={adapter}\n{'=' * 70}", flush=True)
    subprocess.run(train, check=True)
    workspace.commit()

    print(f"\n{'=' * 70}\n[{tag}] eval\n{'=' * 70}", flush=True)
    subprocess.run(
        [
            "python", "-m", "tooluse.eval.run_eval",
            "--tag", tag, "--adapter", output,
            "--n-seeds", str(n_seeds), "--trials", str(trials), "--out", "/work/results",
        ],
        check=True,
    )
    workspace.commit()
    print(f"[{tag}] done", flush=True)


BFCL_ARMS = {
    "base": None,
    "sft": "/work/checkpoints/sft",
    "grpo": "/work/checkpoints/grpo",
    "rl_only_long": "/work/checkpoints/rl_only_long",
    "sft_mixed": "/work/checkpoints/sft_mixed",
    "grpo_mixed": "/work/checkpoints/grpo_mixed",
}


@app.function(volumes=VOLUMES, timeout=1 * HOURS)
def merge_adapters() -> None:
    """Merge each LoRA adapter into full base weights for BFCL.

    BFCL's `--lora-modules` flag does not do what its README implies. It is forwarded to
    `vllm serve`, which registers the adapter, but every request BFCL then sends names the
    *base* model:

        api_response = self.client.completions.create(
            model=self.model_path_or_id,   # always the base path, never a LoRA name

    So the adapter is loaded and never applied. Running the sweep that way would have produced
    four identical copies of the base numbers and a confident "nothing transfers" conclusion,
    which is a worse outcome than getting no numbers at all. Merging sidesteps it entirely:
    each arm becomes a standalone model directory that needs no adapter machinery.
    """
    import shutil
    from pathlib import Path

    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base_id = "Qwen/Qwen3-0.6B"

    # Copy the hub's own config and tokenizer files rather than re-serialising them. This image
    # runs transformers 5.15.1 and BFCL runs 4.51.3, and the two do not round-trip: 5.x writes
    # `extra_special_tokens` as a list, 4.51.3 expects a dict, and loading dies on
    # `'list' object has no attribute 'keys'`. Nothing here modifies the tokenizer, so the
    # upstream files are both correct and version-neutral.
    # Filter by extension explicitly. `allow_patterns` governs what `snapshot_download` *fetches*,
    # not what the snapshot directory contains — loading the base model above already populated
    # it with `model.safetensors`. Copying everything therefore overwrote each freshly merged
    # checkpoint with base weights, and produced four byte-identical "merged" models that BFCL
    # then scored as four indistinguishable arms.
    hub = Path(snapshot_download(base_id, allow_patterns=["*.json", "*.txt", "*.jinja"]))
    sidecars = [
        p
        for p in hub.iterdir()
        if p.is_file() and p.suffix in {".json", ".txt", ".jinja"} and "safetensors" not in p.name
    ]
    print(f"[merge] sidecars: {sorted(p.name for p in sidecars)}", flush=True)

    for tag, adapter in BFCL_ARMS.items():
        if adapter and not Path(adapter).exists():
            print(f"[merge] {tag}: no adapter at {adapter}, skipping", flush=True)
            continue
        target = Path(f"/work/merged/{tag}")
        print(f"\n[merge] {tag} <- {adapter or 'base weights'}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16)
        if adapter:
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        model.save_pretrained(target)
        for path in sidecars:
            shutil.copy2(path, target / path.name)
        workspace.commit()
        print(f"[merge] wrote {target}", flush=True)


@app.function(image=bfcl_image, volumes=VOLUMES, timeout=10 * 60)
def bfcl_probe() -> None:
    """CPU-only check that the merged checkpoints load under BFCL's pinned stack.

    The merged weights were written by transformers 5.15.1 and BFCL's vllm extra forces 4.51.3,
    so config and chat-template compatibility across that gap is worth proving for free rather
    than discovering partway into a paid GPU run.
    """
    from pathlib import Path

    import transformers
    import vllm
    from transformers import AutoConfig, AutoTokenizer

    print(f"vllm {vllm.__version__} | transformers {transformers.__version__}", flush=True)
    print("files:", sorted(p.name for p in Path("/work/merged/base").iterdir()), flush=True)
    config = AutoConfig.from_pretrained("/work/merged/base")
    tokenizer = AutoTokenizer.from_pretrained("/work/merged/base")
    print(f"config ok: {config.model_type} | tokenizer ok: {type(tokenizer).__name__}", flush=True)
    print(f"chat template present: {tokenizer.chat_template is not None}", flush=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "f", "description": "d", "parameters": {}}}],
        tokenize=False,
        add_generation_prompt=True,
    )
    print(f"renders tools: {'<tools>' in rendered or 'f' in rendered}", flush=True)


@app.function(gpu=GPU, volumes=VOLUMES, timeout=5 * HOURS)
def mixed_arm(apigen: int = 500, hermes: int = 500, n_seeds: int = 100, trials: int = 4) -> None:
    """SFT on a corpus that actually contains parallel calls, then evaluate in-domain.

    Tests one hypothesis: BFCL `parallel` collapsing to 0/200 is caused by the *absence* of
    multi-call examples, not by capacity loss or general forgetting. Total trajectories are held
    at 1,000 to match the existing `sft` arm, so the only variable is composition.

    The in-domain eval is here because the mix halves the APIGen share, and APIGen is what makes
    SFT valuable as an RL prior (+0.301, §3.1). Buying parallel calling by destroying the prior
    would be a bad trade, and only measuring both shows it.
    """
    import subprocess

    def stage(name: str, command: list[str]) -> None:
        print(f"\n{'=' * 70}\n[mixed] {name}\n{'=' * 70}", flush=True)
        subprocess.run(command, check=True)
        workspace.commit()
        print(f"[mixed] {name} committed", flush=True)

    stage(
        "1/3 build mixed corpus",
        [
            "python", "-m", "tooluse.data.prepare_sft",
            "--limit", str(apigen),
            "--hermes", str(hermes),
            "--out", "/work/data/sft_mixed.jsonl",
            "--cache", "/cache/apigen-mt_5k.json",
        ],
    )
    stage(
        "2/3 SFT",
        [
            "python", "-m", "tooluse.train.sft",
            "--data", "/work/data/sft_mixed.jsonl",
            "--output", "/work/checkpoints/sft_mixed",
            "--limit", str(apigen + hermes),
            "--max-length", "8192",
        ],
    )
    stage(
        "3/3 eval in-domain",
        [
            "python", "-m", "tooluse.eval.run_eval",
            "--tag", "sft_mixed", "--adapter", "/work/checkpoints/sft_mixed",
            "--n-seeds", str(n_seeds), "--trials", str(trials), "--out", "/work/results",
        ],
    )
    print("\n[mixed] done — run merge_adapters then bfcl_sweep for the external numbers", flush=True)


@app.function(volumes=VOLUMES, timeout=30 * 60)
def diagnose_merge() -> None:
    """Find out why merging the SFT adapter changes no weights."""
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    workspace.reload()
    adapter_weights = load_file("/work/checkpoints/sft/adapter_model.safetensors")
    print(f"[diag] adapter tensors: {len(adapter_weights)}", flush=True)
    for name in list(adapter_weights)[:4]:
        tensor = adapter_weights[name]
        print(f"[diag]   {name} {tuple(tensor.shape)} absmax={tensor.abs().max():.4g}", flush=True)
    lora_b = [v for k, v in adapter_weights.items() if "lora_B" in k]
    print(f"[diag] lora_B tensors: {len(lora_b)}, all zero: {all(t.abs().max() == 0 for t in lora_b)}")

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
    probe = "model.layers.0.self_attn.q_proj.weight"
    before = dict(model.named_parameters())[probe].detach().clone()

    peft_model = PeftModel.from_pretrained(model, "/work/checkpoints/sft")
    active = [n for n, _ in peft_model.named_modules() if "lora_A" in n]
    print(f"[diag] lora modules attached: {len(active)}", flush=True)

    merged = peft_model.merge_and_unload()
    after = dict(merged.named_parameters())[probe].detach()
    delta = (after.float() - before.float()).abs().max().item()
    print(f"[diag] max |Δ| on {probe}: {delta:.6g}", flush=True)
    print(f"[diag] merge {'WORKED' if delta > 0 else 'was a NO-OP'}", flush=True)

    # Does the change survive save_pretrained? Written to /tmp, i.e. container-local disk, so
    # this separates "save drops the merge" from "the volume did not receive the write".
    merged.save_pretrained("/tmp/merge_test")
    reloaded = load_file("/tmp/merge_test/model.safetensors")[probe]
    saved_delta = (reloaded.float() - before.float()).abs().max().item()
    print(f"[diag] max |Δ| after save->reload: {saved_delta:.6g}", flush=True)
    print(f"[diag] save {'PRESERVED the merge' if saved_delta > 0 else 'DROPPED the merge'}", flush=True)

    on_volume = load_file("/work/merged/sft/model.safetensors")[probe]
    volume_delta = (on_volume.float() - before.float()).abs().max().item()
    print(f"[diag] max |Δ| of /work/merged/sft vs base: {volume_delta:.6g}", flush=True)


@app.function(volumes=VOLUMES, timeout=20 * 60)
def verify_merged_differ() -> None:
    """Prove the four merged checkpoints are actually different models.

    BFCL reports every arm as statistically identical to base, and the boring explanation for
    that is a silent merge failure producing four copies of the base weights. Since that would
    invalidate the conclusion rather than support it, it gets checked rather than assumed.
    """
    import hashlib
    from pathlib import Path

    workspace.reload()
    digests = {}
    for tag in BFCL_ARMS:
        weights = Path(f"/work/merged/{tag}") / "model.safetensors"
        if not weights.exists():
            print(f"[verify] {tag}: not merged, skipping", flush=True)
            continue
        digests[tag] = hashlib.sha256(weights.read_bytes()).hexdigest()[:16]
        print(f"[verify] {tag}: {digests[tag]}", flush=True)
    unique = len(set(digests.values()))
    print(f"[verify] {unique}/{len(digests)} distinct — {'OK' if unique == len(digests) else 'MERGE FAILED'}")


@app.function(gpu=GPU, image=bfcl_image, volumes=VOLUMES, timeout=3 * HOURS)
def bfcl(arms: str = "base", categories: str = "simple_python", threads: int = 64) -> dict:
    """Run BFCL on one merged checkpoint. Predictions are pre-registered in BFCL_PREREGISTRATION.md.

    `simple_python` is the smoke category: single-turn and cheap, and if a checkpoint cannot
    register above zero there, the expensive multi-turn categories will be uniformly zero and
    the comparison this experiment exists to make is unmeasurable.

    `threads` is the only throughput knob that matters here, because BFCL caps in-flight
    requests at exactly this value (`ThreadPoolExecutor(max_workers=num_threads)`). At the
    default 8, a request took ~4s and the A10 sat idle: Qwen3-0.6B is 1.2 GB of weights against
    600 GB/s of bandwidth, so decode is nowhere near saturated and ~19 GB is left for KV cache.
    Raising this is free, whereas a larger GPU costs 3.6x (H100) to buy back only per-token
    latency — the wrong lever for a workload that is latency-bound at low concurrency.
    """
    import json
    import subprocess
    from pathlib import Path

    tag = arms
    merged = f"/work/merged/{tag}"
    if not Path(merged).exists():
        raise RuntimeError(f"{merged} missing — run merge_adapters first")
    print(f"\n{'=' * 70}\n[bfcl] {tag}: {categories}\n{'=' * 70}", flush=True)
    subprocess.run(
        [
            "bfcl", "generate",
            "--model", "Qwen/Qwen3-0.6B-FC",
            "--test-category", categories,
            "--backend", "vllm",
            "--local-model-path", merged,
            "--num-gpus", "1",
            "--gpu-memory-utilization", "0.85",
            "--num-threads", str(threads),
            "--result-dir", f"/work/bfcl/{tag}/result",
        ],
        check=True,
    )
    workspace.commit()
    subprocess.run(
        [
            "bfcl", "evaluate",
            "--model", "Qwen/Qwen3-0.6B-FC",
            "--test-category", categories,
            "--result-dir", f"/work/bfcl/{tag}/result",
            "--score-dir", f"/work/bfcl/{tag}/score",
        ],
        check=True,
    )
    workspace.commit()
    # Score files are JSONL whose *first* line is the summary and whose remaining lines are
    # individual failures, so they must be read line-wise rather than with `json.load`.
    scores = {}
    for path in sorted(Path(f"/work/bfcl/{tag}/score").rglob("*.json")):
        head = json.loads(path.read_text().splitlines()[0])
        name = path.stem.replace("BFCL_v4_", "").replace("_score", "")
        scores[name] = head
        print(f"[bfcl] SCORE {tag} {name}: {json.dumps(head)}", flush=True)
    print(f"[bfcl] {tag} done", flush=True)
    return {"tag": tag, "scores": scores}


@app.local_entrypoint()
def bfcl_sweep(jobs: str = "", threads: int = 64) -> None:
    """Fan the sweep out across containers, one arm per GPU.

    The arms are completely independent, so running them in parallel costs the same GPU-seconds
    and divides wall clock by the number of arms. That is a strictly better trade than moving to
    a larger GPU, which would cost 3.6x per hour (H100 vs A10) to buy a much smaller speedup on
    a workload that is not compute-bound.

    `jobs` is a semicolon-separated list of `tag:categories`, so finished work is not repeated.
    """
    import json

    specs = [job.split(":", 1) for job in jobs.split(";") if job.strip()]
    print(f"[sweep] {len(specs)} containers in parallel", flush=True)
    for result in bfcl.starmap([(tag, categories, threads) for tag, categories in specs]):
        print(f"[sweep] {result['tag']}: {json.dumps(result['scores'])}", flush=True)


@app.function(volumes=VOLUMES, timeout=1 * HOURS)
def fetch(path: str) -> bytes:
    """Read a file back out of the workspace volume."""
    from pathlib import Path

    return Path(path).read_bytes()


@app.local_entrypoint()
def main() -> None:
    """The whole loop from an empty workspace, in order."""
    prepare.remote()
    smoke.remote()
    evaluate.remote(tag="base", n_seeds=120)
    finish.remote()


@app.local_entrypoint()
def resume() -> None:
    """Everything except the base eval, for a fresh workspace when base results already exist.

    The base numbers are produced by code that has not changed and by the same held-out
    seeds and decoding settings, so re-measuring them would only spend GPU minutes to
    reproduce a number already in `results/`.
    """
    prepare.remote()
    finish.remote()
