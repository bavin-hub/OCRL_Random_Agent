# Robotic World Model

A learned world model for the Unitree G1 humanoid robot. Given an RGB-D frame and an action, the model predicts the next frame — enabling the robot to "imagine" future states before acting.

---

## Setup

See [setup.md](setup.md) for environment setup.

---

## Architecture

Two world model variants:

- **`wm_gru`** — state-based. Predicts next proprioceptive state from current state + action.
- **`wm_vision_rssm`** — vision-based. Predicts next RGB-D frame from current frame + action + proprioception.

### Vision World Model (`wm_vision_rssm`)

RSSM with a 6-stage ResNet encoder/decoder for 256x256 RGB-D, GRU dynamics, and discrete categorical latents.

```
RGB-D(t) + Action(t) + Prop(t) → Encoder → Prior/Posterior → GRU → Decoder → RGB-D(t+1)
```

Key components:
- **Encoder**: 6-stage ResNet (256→4 spatial, with GroupNorm + residual blocks)
- **Latent**: 32x32 discrete categorical with straight-through gradients
- **Decoder**: U-Net with skip connections from 5 encoder stages
- **Loss**: Symlog MSE (rgb + depth) + LPIPS perceptual + KL with free-bits and balancing

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

### Vision world model

```bash
python main.py \
  --model_type wm_vision_rssm \
  --run_mode train \
  --train_dirs \
    data/pretraining_rollouts/25000_transitions/2026-04-08_13-45-36 \
    data/pretraining_rollouts/25000_transitions/2026-04-08_15-38-39 \
  --split 0.8
```

`--split 0.8` automatically shuffles all `.db` files and splits 80% train / 20% test. The split is deterministic (seed=42).

You can also provide explicit train/test dirs:

```bash
python main.py \
  --model_type wm_vision_rssm \
  --run_mode train \
  --train_dirs data/pretraining_rollouts/run1 data/pretraining_rollouts/run2 \
  --test_dirs  data/pretraining_rollouts/run3
```

### State-based world model

```bash
python main.py \
  --model_type wm_gru \
  --run_mode train \
  --db_dir_name pretraining_rollouts/1000000_transitions
```

---

## Evaluate

```bash
python main.py \
  --model_type wm_vision_rssm \
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
| `batch_size` | 16 | Micro-batch size per GPU forward pass |
| `grad_accum_steps` | 4 | Accumulate gradients over N batches (effective batch = 64) |
| `seq_len` | 40 | GRU unroll length (history depth for training) |
| `epochs` | 300 | Training epochs |
| `lpips_weight` | 0.5 | Perceptual loss weight (VGG-based) |
| `depth_scale` | 50.0 | Depth normalization divisor |

---

## Project Structure

```
├── config.json                  # all hyperparameters
├── main.py                      # entry point (train / eval)
├── models/
│   ├── world_model.py           # state-based GRU world model
│   └── vision_world_model.py    # vision RSSM world model
├── runners/
│   ├── train.py                 # training loops
│   ├── eval.py                  # evaluation
│   └── agent.py                 # dispatcher
├── data/
│   ├── preprocessor.py          # dataset loaders
│   ├── pretraining_rollouts/    # collected rollout data
│   └── baseline_policies/       # RL policies for data collection
├── utils.py                     # save/load/plotting helpers
└── logs/                        # saved models, checkpoints, plots
```

---

## License

[MIT](LICENSE)
