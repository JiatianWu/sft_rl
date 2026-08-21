# tooluse

SFT → GRPO → eval loop for a sub-4B tool-use agent. See [PLAN.md](PLAN.md) for the design,
trade-offs, and kill-gates.

## Local setup

The laptop only authors code and submits Modal jobs, so the local environment is
deliberately torch-free.

```bash
uv sync                  # core + dev, macOS-safe
source .venv/bin/activate
modal setup              # one-time browser auth, writes ~/.modal.toml
```

Use this venv, not another project's. The `train` and `eval` extras are marked
`sys_platform == 'linux'` and will not install locally by design.

## Why the versions are what they are

Every pin traces to a constraint in a producer's own metadata or source, not to a doc:

| Pin | Forced by |
|---|---|
| `vllm==0.26.0` | `trl/import_utils.py` raises unless `0.17.0 <= vllm <= 0.26.0`. 0.27.x fails this check. |
| `torch==2.11.0` | vllm 0.26.0's `requires_dist` pins `torch==2.11.0` exactly. |
| `transformers>=5.5.3` | vllm 0.26.0 requires it; stricter than the `>=5.2.0` that `environment_factory` needs. |
| `trl>=1.9.0` | `environment_factory`, `get_reward`, `max_tool_calling_iterations` appear in the stable `trl/trainer/grpo_trainer.py` at 1.9.0. |
| Python 3.12 | `trl[harbor]` needs `>=3.12`; modal and vllm both cap at `<3.15`. |

Unsloth is **not** in this project. Its current release requires `trl<=0.24.0`,
`transformers<=5.5.0`, and `datasets<4.4.0`, which is mutually exclusive with both
`environment_factory` (needs trl >= 1.9.0) and vllm (needs transformers >= 5.5.3). SFT
therefore uses TRL's `SFTTrainer` + PEFT — PLAN.md's pre-committed T1 fallback — which
keeps SFT, GRPO, and eval on one image and one lockfile.

## Drift guarantee

`infra/app.py` builds its images with `Image.uv_sync()` against this same
`pyproject.toml` + `uv.lock`, so the container cannot silently diverge from the
resolution recorded here. Bump a version by editing `pyproject.toml` and re-running
`uv lock`; never by editing an image definition.
