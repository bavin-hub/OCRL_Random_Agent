# Policy Training Directory

This directory contains the necessary wrappers and scripts to run Reinforcement Learning (RL) policy training using either the standard simulation baseline (MuJoCo) or the mathematical PyTorch **World Model**.

## Files
- `world_model_env.py`: Contains the `WorldModelEnv` class, which serves as a seamless bridge between `RSL-RL` running policies and the PyTorch-based GRU World Model. It satisfies the core requirements of `RslRlVecEnvWrapper` by handling batched sequence rollouts (`reset`) and kinematic tracking reward evaluation natively in PyTorch math (`step`).
- `train.py`: The main entry point for kicking off PPO (RSL-RL) training loops.
- `local_env/`: Local environment overrides with modified reward functions. Registers `Unitree-G1-Rough-MJ` and `Unitree-G1-Flat-MJ` tasks that patch the upstream submodule configs without requiring changes to `unitree_rl_mjlab`.
  - `rewards.py`: Modified reward functions (see [Reward Changes](#reward-changes) below).
  - `__init__.py`: Imports G1 configs from the submodule, patches reward functions, and registers the MJ task variants.

## How it works
Both the World Model mode and Physics mode utilize identical reward equations based **only on kinematics** (e.g. tracking velocity, maintaining pose bounds, penalizing angular momentum). This ensures we can train completely parallel policies under an identical structure to evaluate how well the World Model captures purely mathematical environment dynamics.

## Available Tasks

| Task ID | Description |
|---|---|
| `Unitree-G1-Rough-MJ` | G1 on rough terrain with modified rewards |
| `Unitree-G1-Flat-MJ` | G1 on flat terrain with modified rewards |

### Running with Physics (Baseline)
Launch the standard simulation wrapper using MuJoCo:
```bash
python policy_training/train.py Unitree-G1-Flat-MJ
```

### Running with the World Model (MBRL)
Swap out the physics for the batched GRU trajectory predictor by adding `--use-world-model`:
```bash
python policy_training/train.py Unitree-G1-Flat-MJ \
    --use-world-model \
    --world-model-db-dir pretraining_rollouts/1000000_transitions \
    --world-model-checkpoint /path/to/logs/saved_models/wm_gru_.../wm_gru-epoch_30.pth
```

## Reward Changes

The `local_env/rewards.py` file contains the following modifications over the upstream submodule rewards:

| Reward Function | Change |
|---|---|
| `track_linear_velocity` | Removed 2x multiplier on z-axis velocity error (`xy_error + z_error` instead of `xy_error + 2*z_error`) |
| `track_angular_velocity` | Removed xy angular velocity penalty; only penalizes z-axis tracking error |
| `feet_air_time` | Default threshold lowered from 0.4 to 0.3 |
| `feet_clearance` | Uses `torch.mean` instead of `torch.sum` for cost aggregation; target height set to 0.12 |

Additionally, a `foot_air_time` reward term (weight=1.0) is added to the env config.

### Notes on `reset()`
Because the GRU utilizes an $M=32$ horizon burn-in, calling `reset()` effectively samples an initial chunk directly from the database defined in `--world-model-db-dir` uniformly at random. It streams these 32 states through the GRU to warm up the `ht` state space matrix seamlessly for the next prediction sequence.
