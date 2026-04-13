"""Local velocity env with kinematics-only rewards for G1.

Registers tasks "Unitree-G1-Rough-MJ" and "Unitree-G1-Flat-MJ" that use
the original submodule configs but strip all sensor-dependent rewards,
keeping only kinematics-based rewards computable by both MuJoCo and WorldModelEnv.

Reward parity between the two modes:
  1. track_linear_velocity  (modified: no 2x z penalty)   w= 1.0
  2. track_angular_velocity (modified: z-axis only)        w= 1.0
  3. body_orientation_l2                                   w=-1.0
  4. body_ang_vel                                          w=-0.05
  5. action_rate_l2                                        w=-0.05
  6. joint_acc_l2                                          w=-2.5e-7
  7. joint_pos_limits                                      w=-10.0
  8. pose (variable_posture)                               w= 1.0
  9. stand_still                                           w=-1.0
 10. is_terminated                                         w=-200.0

IMPORTANT: Import this AFTER src.tasks so these registrations
override the originals.
"""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.velocity.config.g1.env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)
from src.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from policy_training.local_env import rewards as local_rewards

# Rewards that require sensors/contact data — removed for parity with world model.
_SENSOR_DEPENDENT_REWARDS = [
  "foot_gait",
  "foot_clearance",
  "foot_slip",
  "soft_landing",
  "self_collisions",
  "angular_momentum",
]


def _patch_rewards(cfg):
  """Strip sensor-dependent rewards and patch kinematics rewards."""
  for key in _SENSOR_DEPENDENT_REWARDS:
    cfg.rewards.pop(key, None)

  # Patch the two modified rewards (bca7b37 changes).
  cfg.rewards["track_linear_velocity"].func = local_rewards.track_linear_velocity
  cfg.rewards["track_angular_velocity"].func = local_rewards.track_angular_velocity

  # Remaining rewards kept as-is from the original config:
  #   body_orientation_l2  w=-1.0   (projected gravity)
  #   body_ang_vel         w=-0.05  (angular velocity xy)
  #   action_rate_l2       w=-0.05  (action difference)
  #   joint_acc_l2         w=-2.5e-7 (joint acceleration)
  #   joint_pos_limits     w=-10.0  (soft limit penalty)
  #   pose                 w=1.0    (variable posture)
  #   stand_still          w=-1.0   (default pose when still)
  #   is_terminated        w=-200.0 (termination penalty)

  return cfg


def _g1_rough_mj(play=False):
  cfg = unitree_g1_rough_env_cfg(play=play)
  return _patch_rewards(cfg)


def _g1_flat_mj(play=False):
  cfg = unitree_g1_flat_env_cfg(play=play)
  return _patch_rewards(cfg)


register_mjlab_task(
  task_id="Unitree-G1-Rough-MJ",
  env_cfg=_g1_rough_mj(),
  play_env_cfg=_g1_rough_mj(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-MJ",
  env_cfg=_g1_flat_mj(),
  play_env_cfg=_g1_flat_mj(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
