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

# τ-bench needs a third image, and for once the isolation is free rather than forced: tau2-bench
# installs into its own uv-managed venv under /opt/tau2, so its dependency set and vLLM's never
# meet. vLLM here matches `bfcl_image` because that combination is known to serve Qwen3.
tau2_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .pip_install("vllm==0.8.5", "transformers==4.51.3", "soundfile")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "git clone --depth 1 https://github.com/sierra-research/tau2-bench /opt/tau2",
        # Core only. The voice extra pulls ElevenLabs/Deepgram and needs API keys we do not have,
        # and `retail` is a text domain.
        "cd /opt/tau2 && /root/.local/bin/uv sync",
    )
    .env(
        {
            "HF_HOME": "/cache/hf",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "TAU2_DATA_DIR": "/opt/tau2/data",
            # LiteLLM reaches the local vLLM server through the `hosted_vllm/` prefix, which reads
            # this variable. Using `openai/` instead would work for the agent and then silently
            # redirect the *user simulator* to the same local 0.6B model, quietly replacing the
            # benchmark's user with our own checkpoint.
            "HOSTED_VLLM_API_BASE": "http://localhost:8000/v1",
            "HOSTED_VLLM_API_KEY": "dummy",
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
    include_abstain: bool = False,
) -> None:
    """One RL arm end to end: GRPO then evaluation, in a single container.

    Used for the two experiments the results made worth running — an RL-only arm
    (`adapter="none"`, a fresh LoRA) to find out whether the SFT stage earns its place given
    that it *lowered* success, and a longer run from the SFT adapter to test whether the
    residual failures are undertraining rather than design.

    `include_abstain` adds `irrelevant_request` to the RL mix, whose correct episode makes no tool
    call at all. It is scored separately rather than folded in, because the headline split has to
    keep meaning what it meant for the arms already reported (§3.9).
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
    if include_abstain:
        train += ["--include-abstain"]

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

    if include_abstain:
        # Separate pass, separate tag: the abstention family is not part of the 600-task split,
        # and its topics are held out from training so this measures restraint rather than recall.
        print(f"\n{'=' * 70}\n[{tag}] eval abstention family\n{'=' * 70}", flush=True)
        subprocess.run(
            [
                "python", "-m", "tooluse.eval.run_eval",
                "--tag", f"{tag}_abstain", "--adapter", output,
                "--families", "irrelevant_request",
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
    "grpo_abstain": "/work/checkpoints/grpo_abstain",
    "sft_1500": "/work/checkpoints/sft_1500",
    "grpo_1500": "/work/checkpoints/grpo_1500",
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
def mixed_arm(
    apigen: int = 500,
    hermes: int = 500,
    tag: str = "sft_mixed",
    n_seeds: int = 100,
    trials: int = 4,
) -> None:
    """SFT on a corpus that actually contains parallel calls, then evaluate in-domain.

    Tests one hypothesis: BFCL `parallel` collapsing to 0/200 is caused by the *absence* of
    multi-call examples, not by capacity loss or general forgetting. At the default 500/500 the
    total is held at 1,000 to match the existing `sft` arm, so composition is the only variable.

    That control turned out to be expensive. §3.8 found the substitution costs the whole of SFT's
    value as an RL prior, because the removed APIGen trajectories are what carry retail protocol —
    so `parallel` and the prior appeared to trade directly. But 1,000 was a control, not a budget,
    and `--apigen 1000 --hermes 500` tests whether they trade at all once the cap is lifted.

    `tag` names both the corpus and the checkpoint, so a second composition cannot overwrite the
    first — the arms have to coexist for the comparison to be possible.
    """
    import subprocess

    def stage(name: str, command: list[str]) -> None:
        print(f"\n{'=' * 70}\n[{tag}] {name}\n{'=' * 70}", flush=True)
        subprocess.run(command, check=True)
        workspace.commit()
        print(f"[{tag}] {name} committed", flush=True)

    stage(
        f"1/3 build corpus ({apigen} apigen + {hermes} hermes)",
        [
            "python", "-m", "tooluse.data.prepare_sft",
            "--limit", str(apigen),
            "--hermes", str(hermes),
            "--out", f"/work/data/{tag}.jsonl",
            "--cache", "/cache/apigen-mt_5k.json",
        ],
    )
    stage(
        "2/3 SFT",
        [
            "python", "-m", "tooluse.train.sft",
            "--data", f"/work/data/{tag}.jsonl",
            "--output", f"/work/checkpoints/{tag}",
            "--limit", str(apigen + hermes),
            "--max-length", "8192",
        ],
    )
    stage(
        "3/3 eval in-domain",
        [
            "python", "-m", "tooluse.eval.run_eval",
            "--tag", tag, "--adapter", f"/work/checkpoints/{tag}",
            "--n-seeds", str(n_seeds), "--trials", str(trials), "--out", "/work/results",
        ],
    )
    print(f"\n[{tag}] done — run merge_adapters then bfcl_sweep for the external numbers", flush=True)


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


@app.local_entrypoint()
def tau2_sweep(
    tags: str = "base,sft,grpo_1500",
    num_tasks: int = 10,
    num_trials: int = 1,
    task_ids: str = "",
) -> None:
    """Run several arms against τ-bench in parallel, one container each.

    Every arm shares the one customer endpoint, which is the point: the user simulator has to be
    identical across arms for the comparison to mean anything, and a single endpoint autoscales
    to serve them all.
    """
    import json

    specs = [tag.strip() for tag in tags.split(",") if tag.strip()]
    print(f"[sweep] {len(specs)} arms in parallel: {specs}", flush=True)
    jobs = [
        (tag, num_tasks, num_trials, "openai/Qwen/Qwen3.6-27B-FP8", 4, 100,
         "https://jiatianwuwork--ep-tau2-user-server.us-west.modal.direct/v1", task_ids)
        for tag in specs
    ]
    for result in tau2.starmap(jobs):
        print(f"[sweep] {json.dumps(result)}", flush=True)


def _optional_secret(name: str) -> list:
    """Attach a secret only when it exists.

    Modal resolves every secret the app references when the app *starts*, not when the function
    using it is called, so an absent `openai-key` blocks `tau2_probe` — which deliberately needs no
    credentials. That would defeat the entire point of having a free pre-flight check for the one
    failure mode that silently produces a full sweep of zeros.
    """
    try:
        secret = modal.Secret.from_name(name)
        secret.hydrate()
        return [secret]
    except Exception:
        return []


def _warm_user_endpoint(user_api_base: str, model: str, minutes: int = 12) -> None:
    """Block until the customer endpoint answers a real request.

    A Modal Endpoint scales to zero, and a 27B takes about two and a half minutes to wake. τ-bench
    gives a failing call four attempts and then records the simulation as an infrastructure error
    with zero messages, so firing several arms at a cold endpoint destroys every simulation in the
    sweep before a single conversation starts. That is exactly how the first pilot died: 30 of 30
    simulations lost to 503s.
    """
    import json
    import time
    import urllib.error
    import urllib.request

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 4,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    deadline = time.time() + minutes * 60
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            request = urllib.request.Request(
                f"{user_api_base}/chat/completions",
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.environ.get('TAU2_USER_API_KEY', '')}",
                },
            )
            urllib.request.urlopen(request, timeout=120).read()
            print(f"[tau2] customer endpoint warm after {attempt} attempt(s)", flush=True)
            return
        except Exception as error:  # noqa: BLE001
            print(f"[tau2] warming customer, attempt {attempt}: {type(error).__name__}", flush=True)
            time.sleep(20)
    raise RuntimeError(f"customer endpoint still cold after {minutes} minutes")


def _serve_vllm(tag: str, port: int = 8000):
    """Start vLLM's OpenAI server on a merged checkpoint and wait until it answers.

    `--tool-call-parser hermes` is the load-bearing flag. Qwen3 emits tool calls as
    `<tool_call>{...}</tool_call>` inside the message body; without a parser vLLM returns that
    verbatim as `content` and `tool_calls` stays empty. τ-bench would then record an agent that
    never called a tool, score 0 on all 114 retail tasks, and look exactly like "0.6B is too small
    for this benchmark" — a conclusion that is cheap to reach, expensive to unwind, and wrong.
    `tau2_probe` exists to fail on this before any paid API call is made.
    """
    import subprocess
    import time
    import urllib.request

    server = subprocess.Popen(
        [
            "vllm", "serve", f"/work/merged/{tag}",
            "--served-model-name", tag,
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--port", str(port),
            "--gpu-memory-utilization", "0.85",
            # 16384 was not enough: retail conversations grow past it and litellm raises
            # ContextWindowExceeded, which τ-bench records as an infra error and drops from the
            # denominator. Qwen3 handles 32768 natively.
            "--max-model-len", "32768",
        ]
    )
    for _ in range(180):
        if server.poll() is not None:
            raise RuntimeError(f"vllm exited with {server.returncode} before becoming ready")
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            print(f"[tau2] vllm serving {tag}", flush=True)
            return server
        except Exception:
            time.sleep(5)
    server.terminate()
    raise RuntimeError("vllm did not become ready within 15 minutes")


@app.function(gpu=GPU, image=tau2_image, volumes=VOLUMES, timeout=1 * HOURS)
def tau2_probe(tag: str = "grpo_1500") -> None:
    """Assert the served checkpoint actually emits parsed tool calls. No API key needed.

    This is the one trap from the BFCL run that would repeat verbatim here, so it gets checked
    on its own, for free, before the benchmark is allowed to cost anything.
    """
    import json
    import urllib.request

    server = _serve_vllm(tag)
    try:
        payload = {
            "model": tag,
            "messages": [
                {"role": "system", "content": "You are a retail agent. Use the tools provided."},
                {"role": "user", "content": "Cancel my order #W123. My email is amy@example.com."},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "find_user_id_by_email",
                        "description": "Find a user id from their email address.",
                        "parameters": {
                            "type": "object",
                            "properties": {"email": {"type": "string"}},
                            "required": ["email"],
                        },
                    },
                }
            ],
            "temperature": 0.0,
            "max_tokens": 256,
        }
        request = urllib.request.Request(
            "http://localhost:8000/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        message = json.loads(urllib.request.urlopen(request, timeout=120).read())["choices"][0]["message"]
        print(f"[probe] content: {message.get('content')!r}", flush=True)
        print(f"[probe] tool_calls: {json.dumps(message.get('tool_calls'))}", flush=True)

        calls = message.get("tool_calls") or []
        if not calls:
            raise RuntimeError(
                "no parsed tool_calls — τ-bench would score every task 0 and it would look like "
                "the model is simply too small. Check --tool-call-parser."
            )
        print(f"[probe] OK — {calls[0]['function']['name']} parsed as a tool call", flush=True)
    finally:
        server.terminate()


@app.function(
    image=tau2_image,
    timeout=1 * HOURS,
    secrets=_optional_secret("tau2-user-token"),
)
def tau2_user_probe(
    user_api_base: str = "https://jiatianwuwork--ep-tau2-user-server.us-west.modal.direct/v1",
    model: str = "Qwen/Qwen3.6-27B-FP8",
) -> None:
    """Check the customer endpoint answers, and answers *in character*, before paying for a run.

    A user simulator that cannot stay in role is the one failure this benchmark cannot survive:
    TAU2_PREREGISTRATION.md records that a weak customer penalises asking far more than acting,
    which is the exact axis P18 measures. The 0.6B dry run failed here by replying "I'm here to
    assist you with your request" — playing agent instead of customer — so the probe checks the
    reply looks like a customer rather than merely checking for a 200.
    """
    import json
    import os
    import urllib.request

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a customer contacting retail support. You want to cancel order "
                    "#W123. Your email is amy@example.com. Stay in character as the customer. "
                    "Reveal details only when asked."
                ),
            },
            {"role": "assistant", "content": "Hi! How can I help you today?"},
        ],
        "temperature": 0.0,
        "max_tokens": 120,
        # Qwen3.6 thinks by default and spent the entire 120-token budget on reasoning, returning
        # empty content with finish_reason=length. Every customer turn would have been blank, which
        # τ-bench would have scored as the agent failing to make progress.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    import time

    def call() -> dict:
        request = urllib.request.Request(
            f"{user_api_base}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('TAU2_USER_API_KEY', '')}",
            },
        )
        return json.loads(urllib.request.urlopen(request, timeout=300).read())

    # A scaled-to-zero 27B answers 503 until it finishes loading. Retry rather than report a cold
    # start as a broken endpoint; re-raise the body, since urllib's HTTPError does not survive
    # Modal's exception pickling and would otherwise surface as an unrelated SerializationError.
    reply, last = None, ""
    for attempt in range(20):
        try:
            reply = call()
            break
        except urllib.error.HTTPError as error:
            last = f"HTTP {error.code}: {error.read()[:300]!r}"
            print(f"[user-probe] attempt {attempt + 1}: {last}", flush=True)
            time.sleep(30)
        except Exception as error:  # noqa: BLE001
            last = f"{type(error).__name__}: {error}"
            print(f"[user-probe] attempt {attempt + 1}: {last}", flush=True)
            time.sleep(30)
    if reply is None:
        raise RuntimeError(f"customer endpoint never became ready: {last}")

    choice = reply["choices"][0]
    content = choice["message"].get("content") or ""
    print(f"[user-probe] finish_reason: {choice.get('finish_reason')}", flush=True)
    print(f"[user-probe] message keys: {list(choice['message'].keys())}", flush=True)
    print(f"[user-probe] reasoning: {str(choice['message'].get('reasoning_content'))[:200]!r}", flush=True)
    print(f"[user-probe] customer says: {content!r}", flush=True)
    print(f"[user-probe] usage: {reply.get('usage')}", flush=True)

    # Checked first, and deliberately: an empty reply passes a naive in-character test, because a
    # string containing nothing contains no agent-speak either. A customer that says nothing stalls
    # every conversation and would be scored as agent failure.
    if not content.strip():
        raise RuntimeError("customer returned empty content — conversations would stall")

    agentish = ("how can i help", "how may i assist", "i'm here to assist", "happy to help")
    if any(phrase in content.lower() for phrase in agentish):
        raise RuntimeError("customer is answering like an agent — the P18 comparison would be void")
    print("[user-probe] OK — stayed in character", flush=True)


@app.function(image=tau2_image, timeout=1 * HOURS)
def tau2_check() -> None:
    """Verify the τ-bench data set and CLI surface without starting a GPU."""
    import subprocess

    for command in (["tau2", "check-data"], ["tau2", "run", "--help"]):
        subprocess.run(["/root/.local/bin/uv", "run", *command], cwd="/opt/tau2", check=False)


@app.function(
    gpu=GPU,
    image=tau2_image,
    volumes=VOLUMES,
    timeout=4 * HOURS,
    secrets=_optional_secret("openai-key") + _optional_secret("tau2-user-token"),
)
def tau2(
    tag: str = "grpo_1500",
    num_tasks: int = 10,
    num_trials: int = 1,
    user_llm: str = "openai/Qwen/Qwen3.6-27B-FP8",
    concurrency: int = 4,
    max_steps: int = 100,
    user_api_base: str = "https://jiatianwuwork--ep-tau2-user-server.us-west.modal.direct/v1",
    task_ids: str = "",
) -> dict:
    """Run τ-bench retail against one merged checkpoint.

    Why this benchmark, given BFCL already said the gains do not transfer: τ-bench retail is the
    thing `tau-retail-lite` is a simplification *of*, so it is not an independent capability check
    the way BFCL was — it tests whether the simplification was faithful. Two things make that worth
    paying for. It has a user simulator, whose absence is the stated cause of §3.1's headline result
    (SFT scoring 0.035 because APIGen-MT teaches the model to interrogate a user that this
    environment does not have), so it can reverse that finding. And its reward is
    `DB x COMMUNICATE` — a state check times an output check — which is structurally the same
    product `compute_reward` uses, arrived at independently.

    `num_tasks` defaults to a pilot-sized 10. The full retail split is 114 tasks and a 0.6B model
    may well score zero on all of them, which would be the BFCL multi-turn power problem again but
    with a metered API attached, so the pilot decides whether the full run is worth buying.

    Passing `user_llm="hosted_vllm/<tag>"` points the customer at the same local server as the
    agent, which exercises the real `user_simulator` path for free. Useful as a plumbing check;
    useless as a measurement, since a 0.6B customer cannot hold up its half of the conversation.
    (`--user dummy_user` cannot serve this purpose: it exists only for solo mode, which asserts on
    tasks carrying a `ticket` field, and no retail task has one.)

    `user_api_base` routes the customer to a separate OpenAI-compatible server — a Modal Endpoint,
    say — while the agent stays on localhost. It has to be passed per-call rather than through the
    environment because LiteLLM resolves `hosted_vllm/` against a single global
    `HOSTED_VLLM_API_BASE`; without the override both roles land on the same server and the
    benchmark quietly swaps its customer for our own checkpoint.
    """
    import json
    import shutil
    import subprocess
    from pathlib import Path

    if user_api_base:
        _warm_user_endpoint(user_api_base, user_llm.split("/", 1)[1])

    server = _serve_vllm(tag)
    run_name = f"{tag}_retail"
    try:
        command = [
            "/root/.local/bin/uv", "run", "tau2", "run",
            "--domain", "retail",
            "--agent-llm", f"hosted_vllm/{tag}",
            "--user-llm", user_llm,
            "--num-trials", str(num_trials),
            "--max-concurrency", str(concurrency),
            "--max-steps", str(max_steps),
            "--save-to", run_name,
        ]
        # The key travels in the environment, never on argv. A Modal proxy token is
        # `wk-<id>.ws-<secret>`, usable verbatim as an OpenAI bearer key — and when this subprocess
        # fails, `CalledProcessError` prints the whole command line into the logs, which is how an
        # earlier run leaked one. LiteLLM reads `OPENAI_API_KEY` for the `openai/` provider, and the
        # agent is unaffected because it resolves through `hosted_vllm/`.
        environment = dict(os.environ)
        if user_api_base:
            environment["OPENAI_API_KEY"] = os.environ.get("TAU2_USER_API_KEY", "dummy")
            command += [
                "--user-llm-args",
                json.dumps(
                    {
                        "api_base": user_api_base,
                        # Without this the customer spends its whole token budget on reasoning and
                        # returns empty content, stalling every conversation. Verified by
                        # `tau2_user_probe`, which now rejects an empty reply outright.
                        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                        # Survive a mid-run scale-down; the default gives up after four tries,
                        # which is not enough for a cold 27B.
                        "num_retries": 8,
                        "timeout": 300,
                    }
                ),
            ]
        if task_ids:
            # Explicit ids rather than a count, so P22 runs on tasks the pilot never touched.
            # Empties are stripped because a trailing separator (`seq -s,` produces one) reaches
            # τ-bench as an id of "" and aborts the whole run with "Not all tasks were found".
            command += ["--task-ids", *(t for t in task_ids.split(",") if t.strip())]
        elif num_tasks:
            command += ["--num-tasks", str(num_tasks)]
        print(f"\n{'=' * 70}\n[tau2] {tag}: {num_tasks or 'all'} tasks x {num_trials}\n{'=' * 70}", flush=True)
        subprocess.run(command, cwd="/opt/tau2", check=True, env=environment)
    finally:
        server.terminate()

    source = Path(f"/opt/tau2/data/simulations/{run_name}")
    destination = Path(f"/work/tau2/{tag}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    workspace.commit()

    # The raw results are already safely on the volume by this point. Summarising is a convenience,
    # so a schema guess that turns out wrong must not raise and discard a metered run.
    try:
        results = json.loads((destination / "results.json").read_text())
        simulations = results.get("simulations", [])

        # A `reward` of 0.0 has two entirely different meanings here, and conflating them would
        # manufacture a clean "pass^1 = 0.000" that reads as a capability verdict but is really a
        # harness artifact. When a run ends on `max_steps` or an infra error, τ-bench never
        # evaluates it: `db_check`, `communicate_checks` and `reward_basis` all come back null and
        # `reward` defaults to zero. Only simulations that actually reached an evaluator are scored
        # below; the rest are counted and reported separately.
        terminations: dict[str, int] = {}
        scored, rewards = [], []
        for simulation in simulations:
            reason = simulation.get("termination_reason") or "unknown"
            terminations[reason] = terminations.get(reason, 0) + 1
            info = simulation.get("reward_info") or {}
            if info.get("reward_basis") is None:
                continue
            scored.append(simulation)
            rewards.append(info.get("reward", 0.0))

        # TAU2_PREREGISTRATION.md commits to reporting these apart: `sft` could gain on
        # COMMUNICATE alone by being chattier, which is not the §3.1 reversal it would resemble.
        breakdown: dict[str, list[float]] = {}
        for simulation in scored:
            for key, value in ((simulation.get("reward_info") or {}).get("reward_breakdown") or {}).items():
                if isinstance(value, (int, float)):
                    breakdown.setdefault(str(key), []).append(float(value))

        summary = {
            "tag": tag,
            "user_llm": user_llm,
            "n_simulations": len(simulations),
            "n_scored": len(rewards),
            "pass^1": sum(rewards) / len(rewards) if rewards else None,
            "solved": sum(1 for r in rewards if r >= 1.0),
            "components": {k: sum(v) / len(v) for k, v in breakdown.items() if v},
            "terminations": terminations,
            "user_cost": sum(s.get("user_cost") or 0.0 for s in simulations),
        }
    except Exception as error:  # noqa: BLE001 - never lose a paid run to a parse error
        summary = {"tag": tag, "error": repr(error), "raw": str(destination)}
    print(f"[tau2] {json.dumps(summary)}", flush=True)
    return summary


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
