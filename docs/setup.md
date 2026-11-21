# Setup and commands

Run commands from the repository root. The supported workflow is `scripts/train.py`
and `scripts/evaluate.py`. The obsolete offline-data and baseline paths have been retired.

## Environment

Python 3.11, Linux, and an NVIDIA GPU are the tested target. Evaluation uses one
environment; training defaults to 256. Start small before increasing parallelism.
The validation machine has an RTX 5090 (32 GB), driver 580.178.04, and PyTorch
2.11.0+cu130. The 8192-environment historical configuration is not a minimum requirement.

```bash
git clone --branch wm_on_the_fly_training https://github.com/bavin-hub/OCRL_Random_Agent.git
cd OCRL_Random_Agent
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e .
python scripts/check_environment.py
nvidia-smi
```

Use a fresh environment. The original `random_agent` environment contained modified
MJLab and RSL-RL files. The validation runs use clean wheel copies of the pinned
versions instead. Use the published dependencies without modifying `site-packages`. No baseline
submodule is required.

If CUDA is missing only inside a sandbox, enable GPU device access for execution;
reinstalling a working host driver does not fix sandbox isolation.

## Checkpoints

Historical checkpoints are excluded from Git. Their exact local paths and SHA-256
hashes are in [checkpoints.json](checkpoints.json). A public download is not yet
published. Existing local users can use the paths below; a fresh clone can train
from scratch without them.

## Evaluate, record, and plot

Use a new output directory for each invocation; existing result directories are
never overwritten. `--command` is a fixed body-frame velocity command `(vx, vy, yaw_rate)`.
The evaluation disables observation noise, pushes, and command curriculum, but retains
startup/reset randomization. It reports the simulator reward, not the custom training reward.

```bash
MUJOCO_GL=egl python scripts/evaluate.py \
  --checkpoint artifacts/historical/saved_models/model_3710.pt \
  --world-model artifacts/historical/logs/ckpts/wm_gru_2026-05-09_23:10:44/wm-itr-ckpt-epoch_3710.pth \
  --steps 1000 --seed 42 --command 0.5 0 0 \
  --prediction-horizon 8 --video --output runs/eval-3710
python scripts/plot_results.py runs/eval-3710
```

Outputs: `rollout.mp4`, `poster.png`, `metrics.csv`, `summary.json`, `manifest.json`,
and (when an uninterrupted trajectory is long enough) `predictions.npz` and forecast
plots. Eight prediction steps span 0.16 seconds after 32 history steps; the model's
original default evaluation horizon is 24 steps. Missing forecast data is reported
explicitly rather than spanning episode resets.

Compare the earlier run with the same seed and protocol:

```bash
MUJOCO_GL=egl python scripts/evaluate.py \
  --checkpoint artifacts/historical/saved_models_copy/model_7030.pt \
  --world-model artifacts/historical/logs/ckpts/wm_gru_2026-05-09_19:02:34/wm-itr-ckpt-epoch_7030.pth \
  --steps 1000 --seed 42 --command 0.5 0 0 \
  --prediction-horizon 8 --video --output runs/eval-7030
python scripts/plot_results.py runs/eval-7030
```

Interactive viewing uses the same policy observation adapter:

```bash
python scripts/evaluate.py --viewer native --output runs/view-3710
# Use --viewer viser for browser viewing; omit --video in interactive mode.
```

`--action-history scaled` (default) reproduces the historical real-rollout convention.
`--action-history normalized` probes the convention used by historical imagination.
Always record this option when comparing results; it changes policy inputs.

## Train from scratch

```bash
python scripts/train.py --num-envs 256 --iterations 1000 \
  --warmup 250 --seed 42 --output runs/train-01
python scripts/plot_results.py runs/train-01
```

Small end-to-end smoke test, including imagination updates and saving:

```bash
python scripts/train.py --num-envs 8 --iterations 4 --warmup 1 \
  --mini-batch-size 8 --world-model-batches 1 --save-interval 1 \
  --output runs/train-smoke
```

Initialize a new experiment from both checkpoints:

```bash
python scripts/train.py --num-envs 256 --iterations 1000 \
  --policy-checkpoint artifacts/historical/saved_models/model_3710.pt \
  --world-model artifacts/historical/logs/ckpts/wm_gru_2026-05-09_23:10:44/wm-itr-ckpt-epoch_3710.pth \
  --output runs/finetune-01
```

This restores model and optimizer state, then starts new collection and iteration
numbering. It is **not an exact training resume**: simulator state, replay history,
random-number generator state, and previous rollout counters are not restored.
Use `--no-imagination` for the simulator PPO path. See `--help` for all options.
Training writes the effective configuration, CLI arguments, metrics, and checkpoints
inside its output directory. The historical training algorithm has unresolved
semantic limitations described in [validation.md](validation.md).

## Checks

```bash
python -m unittest discover -s tests -p 'test_core.py' -v
python scripts/check_environment.py
python scripts/verify_checkpoints.py
```

All documented paths are relative to the repository root. Configuration defaults to
`configs/default.json`; override it with `--config`. Installed commands
`ocrl-train` and `ocrl-evaluate` provide the same interfaces as the scripts.

## Simulator-trained locomotion baseline

The separately trained checkpoint 1199 uses standard MJLab actor observations.
It is a baseline demonstration, not an imagination-trained policy from this run.
Its checksum and provenance are in [baselines.json](baselines.json). Checkpoint
files are local release artifacts and are not included in Git.

```bash
MUJOCO_GL=egl python scripts/evaluate.py \
  --observation-mode mjlab --checkpoint artifacts/baselines/g1-1199.pt \
  --steps 1000 --command 0.5 0 0 --video --width 960 --height 544 \
  --camera-azimuth 135 --output runs/baseline-demo
```

`--observation-mode mjlab` skips world-model loading and forecasts, and passes
MJLab's actor observation directly to the baseline policy. Do not use the
world-model observation adapter for these baseline weights.

## Render the pipeline explainer

```bash
python scripts/create_pipeline_video.py \
  --rollout runs/baseline-demo/rollout.mp4 \
  --checkpoint-label "G1 simulator baseline / checkpoint 1199" \
  --output runs/pipeline-explainer.mp4
```

The output is a 67-second, 1280×720 MP4 at 24 fps, with burned-in captions and no
audio, plus chapter stills, SRT captions, and a source manifest. It uses a white background and animated diagram nodes/arrows to show transition
collection, world-model updates, initial simulator policy training, the warm-start
switch to world-model policy training, repeated optimization, and rollout results. The deployment footage is explicitly labeled
as a separate baseline. FFmpeg is supplied through `imageio-ffmpeg`; install the
pinned package dependencies before rendering.

The animation shows the requested intended training sequence. In the current
implementation, the warm-start phase collects transitions and fits the world model
but does not yet update PPO in the simulator. Adding those initial simulator PPO
updates requires a separate training-code change. No training behavior was changed
when revising the video. Pass `--overwrite` to replace a previous rendered video.
