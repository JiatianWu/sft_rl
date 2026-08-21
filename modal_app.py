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
GPU = os.environ.get("TOOLUSE_GPU", "H100")

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
def grpo(steps: int = 120, adapter: str = "/work/checkpoints/sft", protocol_weight: float = 0.3) -> None:
    import subprocess

    command = [
        "python",
        "-m",
        "tooluse.train.grpo",
        "--output",
        "/work/checkpoints/grpo",
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
def evaluate(tag: str, adapter: str | None = None, n_seeds: int = 20, trials: int = 4) -> None:
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


@app.function(volumes=VOLUMES, timeout=1 * HOURS)
def fetch(path: str) -> bytes:
    """Read a file back out of the workspace volume."""
    from pathlib import Path

    return Path(path).read_bytes()


@app.local_entrypoint()
def main() -> None:
    """The whole loop, in order."""
    prepare.remote()
    smoke.remote()
    evaluate.remote(tag="base")
    sft.remote()
    evaluate.remote(tag="sft", adapter="/work/checkpoints/sft")
    grpo.remote()
    evaluate.remote(tag="grpo", adapter="/work/checkpoints/grpo")
