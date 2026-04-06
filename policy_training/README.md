# Policy Training Directory

This directory contains the necessary wrappers and scripts to run Reinforcement Learning (RL) policy training using either the standard simulation baseline (MuJoCo) or the mathematical PyTorch **World Model**.

## Files
- `world_model_env.py`: Contains the `WorldModelEnv` class, which serves as a seamless bridge between `RSL-RL` running policies and the PyTorch-based GRU World Model. It satisfies the core requirements of `RslRlVecEnvWrapper` by handling batched sequence rollouts (`reset`) and kinematic tracking reward evaluation natively in PyTorch math (`step`).
- `train.py`: The main entry point for kicking off PPO (RSL-RL) training loops.

## How it works
Both the World Model mode and Physics mode utilize identical reward equations based **only on kinematics** (e.g. tracking velocity, maintaining pose bounds, penalizing angular momentum). This ensures we can train completely parallel policies under an identical structure to evaluate how well the World Model captures purely mathematical environment dynamics.

### Running with Physics (Baseline)
Launch the standard simulation wrapper using MuJoCo:
```bash
python policy_training/train.py Unitree-G1-Velocity-Flat
```

### Running with the World Model (MBRL)
Swap out the physics for the batched GRU trajectory predictor by adding `--use-world-model`:
```bash
python policy_training/train.py Unitree-G1-Velocity-Flat \
    --use-world-model \
    --world-model-db-dir pretraining_rollouts/1000000_transitions \
    --world-model-checkpoint /path/to/logs/saved_models/wm_gru_.../wm_gru-epoch_30.pth
```

### Notes on `reset()`
Because the GRU utilizes an $M=32$ horizon burn-in, calling `reset()` effectively samples an initial chunk directly from the database defined in `--world-model-db-dir` uniformly at random. It streams these 32 states through the GRU to warm up the `ht` state space matrix seamlessly for the next prediction sequence.
