"""Unitree G1 constants plus a ground plane for standalone viewing/simulation.

This module re-exports everything from :mod:`g1_constants` and adds helpers that
build a full mjlab :class:`~mjlab.scene.scene.Scene` with flat terrain, matching
how training environments attach the robot to a plane (so feet collide instead of
falling through empty space).

Run the viewer (with physics and ground contact)::

  python -m mjlab.asset_zoo.robots.unitree_g1.g1_with_ground

You can also run this file directly (uses absolute imports so it is not limited
to ``python -m``)::

  python /path/to/g1_with_ground.py
"""

from __future__ import annotations

import mujoco

import mjlab.asset_zoo.robots.unitree_g1.g1_constants as g1_constants
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import *  # noqa: F403
from mjlab.scene.scene import Scene, SceneCfg
from mjlab.terrains.terrain_entity import TerrainEntityCfg


def get_g1_scene_cfg(
  *,
  num_envs: int = 1,
  env_spacing: float = 2.0,
  robot_entity_name: str = "robot",
) -> SceneCfg:
  """Scene with checker ground plane + one G1 (same robot config as ``get_g1_robot_cfg``)."""
  return SceneCfg(
    num_envs=num_envs,
    env_spacing=env_spacing,
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={robot_entity_name: g1_constants.get_g1_robot_cfg()},
  )


def compile_g1_with_ground(
  *,
  num_envs: int = 1,
  env_spacing: float = 2.0,
  device: str = "cpu",
  robot_entity_name: str = "robot",
) -> mujoco.MjModel:
  """Compile MuJoCo model: scene base + terrain plane + G1 with full collision/actuators."""
  cfg = get_g1_scene_cfg(
    num_envs=num_envs,
    env_spacing=env_spacing,
    robot_entity_name=robot_entity_name,
  )
  return Scene(cfg, device=device).compile()


if __name__ == "__main__":
  import mujoco.viewer as viewer

  viewer.launch(compile_g1_with_ground())
