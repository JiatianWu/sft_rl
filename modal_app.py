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

app = modal.App(APP_NAME, image=image)

cache = modal.Volume.from_name("tooluse-cache", create_if_missing=True)
workspace = modal.Volume.from_name("tooluse-workspace", create_if_missing=True)
VOLUMES = {"/cache": cache, "/work": workspace}

HOURS = 60 * 60


@app.function(volumes=VOLUMES, timeout=1 * HOURS)
def prepare() -> None:
    """Download and convert the SFT dataset into the shared volume."""
    from pathlib import Path

    from tooluse.data.prepare_sft import build

    build(limit=None, out_path=Path("/work/data/sft_apigen.jsonl"), cache=Path("/cache/apigen-mt_5k.json"))
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
