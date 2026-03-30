"""Passive viewer loop for mjlab G1 on ground (works from any cwd).

Requires ``mjlab`` importable from the active environment (e.g. conda env).

Uses :func:`compile_g1_with_ground` and holds pose by sending **fixed** position
setpoints equal to the joint angles after ``init_state`` reset. (Updating
``ctrl`` from *live* ``qpos`` every step would make ``ctrl - qpos == 0`` and
remove PD stiffness, so the robot would collapse under gravity.)
"""

from __future__ import annotations

import mujoco
import mujoco.viewer
import numpy as np

from mjlab.asset_zoo.robots.unitree_g1.g1_with_ground import compile_g1_with_ground


def _ctrl_targets_for_joint_position_actuators(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
  """Map each joint-type position actuator to its scalar ``qpos`` (hinge/slide)."""
  targets = np.zeros(model.nu, dtype=np.float64)
  for i in range(model.nu):
    if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
      continue
    joint_id = int(model.actuator_trnid[i, 0])
    qadr = int(model.jnt_qposadr[joint_id])
    jt = int(model.jnt_type[joint_id])
    if jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
      targets[i] = data.qpos[qadr]
  return targets


def main() -> None:
  model = compile_g1_with_ground()
  data = mujoco.MjData(model)

  key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
  if key_id < 0:
    raise RuntimeError("Compiled model has no keyframe 'init_state'.")
  mujoco.mj_resetDataKeyframe(model, data, key_id)
  mujoco.mj_forward(model, data)

  # Fixed setpoints = joint angles at reset (same idea as keyframe ctrl).
  ctrl_hold = _ctrl_targets_for_joint_position_actuators(model, data)

  with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
      if model.nu > 0:
        data.ctrl[:] = ctrl_hold
      mujoco.mj_step(model, data)
      viewer.sync()


if __name__ == "__main__":
  main()
