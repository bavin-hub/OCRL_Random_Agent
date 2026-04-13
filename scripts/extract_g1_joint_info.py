"""Extract G1 joint ordering, default positions, and limits from the compiled MJCF.

Run: python scripts/extract_g1_joint_info.py
"""

import mujoco
import numpy as np

from src.assets.robots.unitree_g1.g1_constants import get_spec, HOME_KEYFRAME


def main():
  spec = get_spec()
  model = spec.compile()
  data = mujoco.MjData(model)

  # Reset to init_state keyframe to get default joint positions.
  key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
  if key_id >= 0:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
  mujoco.mj_forward(model, data)

  print(f"Number of joints (nq): {model.nq}")
  print(f"Number of DoFs (nv): {model.nv}")
  print(f"Number of actuators (nu): {model.nu}")
  print()

  # Joint names, default positions, and limits.
  # Skip the root free joint (7 qpos: 3 pos + 4 quat).
  print("=== Joint info (excluding root free joint) ===")
  print(f"{'idx':>3}  {'joint_name':<35}  {'default_pos':>12}  {'limit_lo':>10}  {'limit_hi':>10}")
  print("-" * 80)

  joint_names = []
  default_positions = []
  limits_lo = []
  limits_hi = []

  for j in range(model.njnt):
    jnt_type = model.jnt_type[j]
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    if name is None:
      name = f"joint_{j}"

    # Skip free joints (root).
    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
      print(f"  [skip free joint: {name}]")
      continue

    qadr = model.jnt_qposadr[j]
    vadr = model.jnt_dofadr[j]
    default_pos = data.qpos[qadr]
    lo = model.jnt_range[j, 0]
    hi = model.jnt_range[j, 1]
    has_limit = bool(model.jnt_limited[j])

    joint_names.append(name)
    default_positions.append(float(default_pos))
    limits_lo.append(float(lo) if has_limit else float("-inf"))
    limits_hi.append(float(hi) if has_limit else float("inf"))

    idx = len(joint_names) - 1
    lim_str = f"{lo:10.4f}  {hi:10.4f}" if has_limit else "  unlimited"
    print(f"{idx:3d}  {name:<35}  {default_pos:12.4f}  {lim_str}")

  print()
  print(f"Total hinge/slide joints: {len(joint_names)}")

  # Print as Python lists for copy-paste into world_model_env.py.
  print("\n=== Copy-paste for world_model_env.py ===\n")
  print("G1_JOINT_NAMES = [")
  for n in joint_names:
    print(f'  "{n}",')
  print("]")
  print()
  print(f"G1_DEFAULT_JOINT_POS = {default_positions}")
  print()
  print(f"G1_JOINT_LIMITS_LO = {limits_lo}")
  print()
  print(f"G1_JOINT_LIMITS_HI = {limits_hi}")

  # Also print soft limits (90% of range, matching soft_joint_pos_limit_factor=0.9).
  soft_lo = []
  soft_hi = []
  for lo, hi in zip(limits_lo, limits_hi):
    mid = (lo + hi) / 2
    half_range = (hi - lo) / 2
    soft_lo.append(mid - 0.9 * half_range)
    soft_hi.append(mid + 0.9 * half_range)
  print()
  print(f"G1_SOFT_JOINT_LIMITS_LO = {[round(x, 6) for x in soft_lo]}")
  print()
  print(f"G1_SOFT_JOINT_LIMITS_HI = {[round(x, 6) for x in soft_hi]}")


if __name__ == "__main__":
  main()
