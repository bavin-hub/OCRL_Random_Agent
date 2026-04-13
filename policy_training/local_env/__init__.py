"""Local velocity env with modified reward functions for G1.

Registers tasks "Unitree-G1-Rough" and "Unitree-G1-Flat" with
patched reward functions from policy_training/local_env/rewards.py.

IMPORTANT: Import this AFTER src.tasks so these registrations
override the originals.
"""

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.registry import register_mjlab_task

from src.tasks.velocity.config.g1.env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)
from src.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from policy_training.local_env import rewards as local_rewards


def _patch_rewards(cfg):
  """Patch reward functions with local modified versions."""
  # 1. Patched linear velocity tracking: removed 2x z_error penalty
  cfg.rewards["track_linear_velocity"].func = local_rewards.track_linear_velocity

  # 2. Patched angular velocity tracking: removed xy_error penalty
  cfg.rewards["track_angular_velocity"].func = local_rewards.track_angular_velocity

  # 3. Patched feet clearance: torch.mean instead of torch.sum, target_height 0.12
  cfg.rewards["foot_clearance"].func = local_rewards.feet_clearance
  cfg.rewards["foot_clearance"].params["target_height"] = 0.12

  # 4. Add foot_air_time reward (from bca7b37 changes)
  cfg.rewards["foot_air_time"] = RewardTermCfg(
    func=local_rewards.feet_air_time,
    weight=1.0,
    params={
      "sensor_name": "feet_ground_contact",
      "threshold": 0.3,
      "command_name": "twist",
      "command_threshold": 0.1,
    },
  )

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
