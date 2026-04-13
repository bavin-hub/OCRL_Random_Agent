# Policy Training Directory

This directory contains the necessary wrappers and scripts to run Reinforcement Learning (RL) policy training using either the standard simulation baseline (MuJoCo) or the mathematical PyTorch **World Model**.

## Files
- `world_model_env.py`: Contains the `WorldModelEnv` class, which serves as a seamless bridge between `RSL-RL` running policies and the PyTorch-based GRU World Model. It satisfies the core requirements of `RslRlVecEnvWrapper` by handling batched sequence rollouts (`reset`) and kinematic tracking reward evaluation natively in PyTorch math (`step`).
- `train.py`: The main entry point for kicking off PPO (RSL-RL) training loops.
- `local_env/`: Local environment overrides with kinematics-only rewards. Registers `Unitree-G1-Rough-MJ` and `Unitree-G1-Flat-MJ` tasks that strip sensor-dependent rewards from the upstream config, ensuring identical reward signals between MuJoCo and world model training.
  - `rewards.py`: Modified `track_linear_velocity` and `track_angular_velocity` reward functions.
  - `__init__.py`: Imports G1 configs from the submodule, patches rewards, and registers the MJ task variants.

## How it works
Both the World Model mode and Physics mode utilize **identical reward equations** based **only on kinematics** (e.g. tracking velocity, maintaining pose bounds, penalizing joint acceleration). This ensures we can train completely parallel policies under an identical structure to evaluate how well the World Model captures environment dynamics.

## Available Tasks

| Task ID | Description |
|---|---|
| `Unitree-G1-Rough-MJ` | G1 on rough terrain with kinematics-only rewards |
| `Unitree-G1-Flat-MJ` | G1 on flat terrain with kinematics-only rewards |

### Running with Physics (Baseline)
```bash
python policy_training/train.py Unitree-G1-Flat-MJ
```

### Running with the World Model (MBRL)
```bash
python policy_training/train.py Unitree-G1-Flat-MJ \
    --use-world-model \
    --world-model-db-dir pretraining_rollouts/1000000_transitions \
    --world-model-checkpoint /path/to/logs/saved_models/wm_gru_.../wm_gru-epoch_30.pth
```

## Reward Functions (Identical Across Both Modes)

| # | Reward | Weight | Description |
|---|---|---|---|
| 1 | `track_linear_velocity` | +1.0 | Exp reward for xy velocity tracking + z penalty (no 2x multiplier) |
| 2 | `track_angular_velocity` | +1.0 | Exp reward for z-axis angular velocity tracking only |
| 3 | `body_orientation_l2` | -1.0 | Penalize non-upright orientation via projected gravity |
| 4 | `body_ang_vel` | -0.05 | Penalize excessive xy angular velocity of torso |
| 5 | `action_rate_l2` | -0.05 | Penalize rapid changes in actions |
| 6 | `joint_acc_l2` | -2.5e-7 | Penalize joint accelerations |
| 7 | `joint_pos_limits` | -10.0 | Penalize joints exceeding soft limits (90% of range) |
| 8 | `pose` | +1.0 | Variable posture: speed-dependent tolerance per joint |
| 9 | `stand_still` | -1.0 | Penalize joint deviation from default when command is near zero |
| 10 | `is_terminated` | -200.0 | Penalty on termination |

### Removed (Sensor-Dependent)
The following rewards from the upstream config are **excluded** because they require contact sensors or dynamics data that the world model does not provide:

`foot_gait`, `foot_clearance`, `foot_slip`, `soft_landing`, `self_collisions`, `angular_momentum`

### Notes on `reset()`
Because the GRU utilizes an $M=32$ horizon burn-in, calling `reset()` effectively samples an initial chunk directly from the database defined in `--world-model-db-dir` uniformly at random. It streams these 32 states through the GRU to warm up the `ht` state space matrix seamlessly for the next prediction sequence.
