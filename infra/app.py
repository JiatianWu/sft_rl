"""Modal images plus the stack-verification entrypoints for PLAN.md gate T1.

Both images are built with `uv_sync` against this repo's own `pyproject.toml` and
`uv.lock`. `uv_sync` defaults to `frozen=True`, so a build can never silently
re-resolve: if the lock and the image disagree, the build fails instead of drifting.
It also skips installing the project itself, so editing `src/tooluse` re-mounts the
source without rebuilding the dependency layer.

`uv_project_dir` resolves against the caller's working directory, so run these from
the repo root:

    modal run infra/app.py::verify_train
    modal run infra/app.py::verify_bfcl
"""

import modal

app = modal.App("tooluse")

_base = modal.Image.debian_slim(python_version="3.12")

# SFT + GRPO + in-domain eval all share one image. Dropping Unsloth removed the
# only reason to maintain a second, older training stack alongside this one.
train_image = (
    _base.uv_sync(uv_project_dir=".", extras=["train"])
    .env({"TRL_EXPERIMENTAL_SILENCE": "1"})
    .add_local_python_source("tooluse")
)

# BFCL is deliberately isolated: bfcl-eval pins numpy==1.26.4, which is
# unsatisfiable alongside vllm 0.26.0's opencv-python-headless>=4.13.0. It reaches
# the model over an OpenAI-compatible endpoint rather than importing vllm itself.
bfcl_image = (
    _base.uv_sync(uv_project_dir=".", extras=["eval"])
    .add_local_python_source("tooluse")
)


@app.function(image=train_image, gpu="A10G", timeout=900)
def verify_train() -> dict:
    """Assert the training stack is internally consistent before spending GPU hours."""
    import inspect

    import torch
    import transformers
    import trl
    import vllm
    from trl import GRPOConfig, GRPOTrainer
    from trl.import_utils import is_vllm_available

    trainer_params = inspect.signature(GRPOTrainer.__init__).parameters
    config_fields = {f.name for f in GRPOConfig.__dataclass_fields__.values()}

    info = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "vllm": vllm.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        # The three assumptions PLAN.md §7 rests on, checked against installed code.
        "has_environment_factory": "environment_factory" in trainer_params,
        "has_max_tool_calling_iterations": "max_tool_calling_iterations" in config_fields,
        "vllm_within_trl_support_window": is_vllm_available(),
    }

    failures = [k for k in (
        "cuda_available",
        "has_environment_factory",
        "has_max_tool_calling_iterations",
        "vllm_within_trl_support_window",
    ) if not info[k]]
    info["ok"] = not failures
    info["failures"] = failures
    return info


@app.function(image=bfcl_image, timeout=900)
def verify_bfcl() -> dict:
    """Confirm BFCL imports and that its numpy pin survived the separate resolution."""
    from importlib.metadata import version

    import numpy

    return {
        "bfcl_eval": version("bfcl-eval"),
        "numpy": numpy.__version__,
        "ok": numpy.__version__.startswith("1.26"),
    }


@app.local_entrypoint()
def main() -> None:
    train = verify_train.remote()
    bfcl = verify_bfcl.remote()
    for name, result in (("train", train), ("bfcl", bfcl)):
        status = "OK" if result.pop("ok") else "FAIL"
        print(f"[{status}] {name}")
        for key, value in result.items():
            print(f"    {key}: {value}")
