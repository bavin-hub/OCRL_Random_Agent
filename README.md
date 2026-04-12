# Robotic World Model

A learned world model for the Unitree G1 humanoid robot. Given an RGB-D frame and an action, the model predicts the next frame — enabling the robot to "imagine" future states before acting.

---

## Setup

See [setup.md](setup.md) for environment setup.

---

## Architecture

### Vision World Model (`wm_vision`)

Deterministic GRU-based world model with separate CNN encoder/decoder branches for RGB and depth.

```
RGB(t) + Depth(t) + Action(t) + Prop(t)
  → CNN Encoders → Fused Embedding → GRU → Latent Head
  → RGB Decoder → RGB(t+1)
  → Depth Decoder → Depth(t+1)
```

Key components:
- **Encoders**: Separate multi-stage ResNet CNNs for RGB (3-ch) and depth (1-ch), each producing a `cnn_embed_dim` vector
- **Dynamics**: Single-layer GRU over fused (RGB + depth + action + proprio) embeddings
- **Latent**: Deterministic MLP bottleneck (no categorical / VQ / KL)
- **Decoders**: Separate CNN decoders for RGB and depth, each receiving half the latent vector
- **Loss**: Symlog MSE on RGB + depth (no KL, no perceptual loss)

### Data normalization
- **RGB**: raw `[0, 255]` images are auto-scaled to `[0, 1]` by the data loader
- **Depth**: raw meters divided by `depth_scale` (default 50.0) in the training loop

---

## Collect Rollouts

Rollouts are collected by running a trained policy in MuJoCo with a simulated D435 camera. Each trajectory saves `.db` files (proprioception + actions) and `.npy` files (RGB-D frames).

### Velocity policy

```bash
cd data/baseline_policies/unitree_rl_mjlab

VISION_CAMERA=robot/d435 VISION_WIDTH=256 VISION_HEIGHT=256 \
VISION_CAPTURE_EVERY_N=1 ROLLOUT_NUM_TRAJECTORIES=100 ROLLOUT_TRANSITIONS_PER_TRAJ=200 \
python scripts/play.py Unitree-G1-Flat-With-Terrain \
  --checkpoint_file=logs/rsl_rl/g1_velocity/2026-03-28_18-01-31/model_900.pt
```

### Tracking policy

Remember to match csv converted motion file(npz) and .pt

```bash
cd data/baseline_policies/unitree_rl_mjlab

VISION_CAMERA=robot/d435 VISION_WIDTH=256 VISION_HEIGHT=256 \
VISION_CAPTURE_EVERY_N=1 ROLLOUT_NUM_TRAJECTORIES=50 ROLLOUT_TRANSITIONS_PER_TRAJ=500 \
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation-With-Terrain \
  --motion_file=src/assets/motions/g1/sprint1_subject2.npz \
  --checkpoint_file=logs/rsl_rl/g1_tracking/2026-04-08_00-53-59/model_9500.pt
```

Rollouts are saved under `data/pretraining_rollouts/`.

---

## Train

```bash
python main.py \
  --model_type wm_vision \
  --run_mode train \
  --train_dirs \
    data/pretraining_rollouts/25000_transitions/2026-04-08_13-45-36 \
    data/pretraining_rollouts/25000_transitions/2026-04-08_15-38-39 \
  --split 0.8
```

### Train/test split

- **`--split 0.8`** automatically shuffles all `.db` files and splits 80% train / 20% test (deterministic, seed=42).
- **`--split_seed N`** changes the shuffle seed.
- **`--test_dirs`** overrides `--split` with explicit test directories:

```bash
python main.py \
  --model_type wm_vision \
  --run_mode train \
  --train_dirs data/pretraining_rollouts/run1 data/pretraining_rollouts/run2 \
  --test_dirs  data/pretraining_rollouts/run3
```

### With W&B logging

Each continued line must end with `\` (including the line before `--wandb_project`), or the shell will stop the command early.
```bash
python main.py \
  --model_type wm_vision \
  --run_mode train \
  --train_dirs \
    data/pretraining_rollouts/25000_transitions/2026-04-08_13-45-36 \
    data/pretraining_rollouts/25000_transitions/2026-04-08_15-38-39 \
  --split 0.8 \
  --wandb_project random_agent \
  --wandb_run_name run2
```

### Resume from checkpoint

```bash
python main.py \
  --model_type wm_vision \
  --run_mode train \
  --train_dirs data/pretraining_rollouts/run1 \
  --load_ckpts_dir <ckpt_dir> \
  --ckpt_name <ckpt_file.pth>
```

---

## Evaluate

```bash
python main.py \
  --model_type wm_vision \
  --run_mode eval \
  --train_dirs data/pretraining_rollouts/25000_transitions/2026-04-08_13-45-36 \
  --model_dir_name <model_dir> \
  --model_name <model_name.pth>
```

---

## Config

Training hyperparameters live in `config.json`. Key vision model settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_height` / `image_width` | 128 | Input resolution (resized from collected frames) |
| `batch_size` | 16 | Micro-batch size per forward pass |
| `grad_accum_steps` | 4 | Accumulate gradients over N batches (effective batch = 64) |
| `seq_len` | 40 | GRU unroll length per training sample |
| `epochs` | 70 | Training epochs |
| `learning_rate` | 1e-4 | Peak learning rate (after warmup) |
| `depth_scale` | 50.0 | Depth normalization divisor |
| `rgb_loss_weight` | 1.0 | RGB loss multiplier |
| `depth_loss_weight` | 1.0 | Depth loss multiplier |
| `grad_clip_norm` | 10.0 | Max gradient norm for clipping |
| `augment` | true | Random horizontal flip + brightness/contrast jitter |

CLI-only flags (not in config.json):

| Flag | Default | Description |
|------|---------|-------------|
| `--wandb_project` | `""` | W&B project name. Empty = disabled |
| `--wandb_run_name` | `""` | Run name (shows in W&B UI) |
| `--wandb_entity` | `""` | W&B team/entity |
| `--split` | `1.0` | Train fraction for auto train/test split |
| `--split_seed` | `42` | Random seed for the split |

---

## Project Structure

```
├── config.json                  # all hyperparameters
├── main.py                      # entry point (train / eval)
├── models/
│   ├── vision_world_model_2.py  # vision world model (deterministic GRU)
│   └── world_model.py           # state-based GRU world model
├── runners/
│   ├── train.py                 # training loop
│   ├── eval.py                  # evaluation
│   └── agent.py                 # dispatcher
├── data/
│   ├── preprocessor.py          # dataset loaders
│   ├── pretraining_rollouts/    # collected rollout data
│   └── baseline_policies/       # RL policies for data collection
├── tools/
│   └── vision_wm_openloop.py    # open-loop rollout visualization
├── utils.py                     # save/load/plotting helpers
└── logs/                        # saved models, checkpoints
```

---

## License

[MIT](LICENSE)
