#!/usr/bin/env python3
"""Batched Unitree G1 velocity env on GPU (default 4096 worlds).

Run from any directory; requires mjlab + PyTorch + CUDA + MuJoCo Warp in the env::

  python /path/to/run_g1_4096_gpu.py
  python /path/to/run_g1_4096_gpu.py --num-envs 2048 --steps 50

Uses the same task definition as ``Mjlab-Velocity-Flat-Unitree-G1`` (flat ground,
full MDP), only overrides ``scene.num_envs`` and ``device``.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg


def main() -> None:
  parser = argparse.ArgumentParser(description="Run batched G1 velocity env on GPU.")
  parser.add_argument("--num-envs", type=int, default=4096, help="Parallel worlds (default 4096).")
  parser.add_argument("--steps", type=int, default=100, help="Environment steps after reset.")
  parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Torch/MJWarp device (default cuda).",
  )
  parser.add_argument(
    "--action-noise",
    type=float,
    default=0.0,
    help="Std dev of Gaussian noise added to actions (default 0 = zeros).",
  )
  args = parser.parse_args()

  if args.device.startswith("cuda") and not torch.cuda.is_available():
    print("CUDA is not available; use --device cpu (4096 envs on CPU will be very slow).", file=sys.stderr)
    sys.exit(1)

  cfg = unitree_g1_flat_env_cfg()
  cfg.scene.num_envs = args.num_envs

  print(f"Building {args.num_envs} envs on {args.device} (first launch compiles GPU kernels)...")
  t0 = time.perf_counter()
  env = ManagerBasedRlEnv(cfg, device=args.device, render_mode=None)
  t_build = time.perf_counter() - t0
  print(f"Built in {t_build:.2f}s  |  action_space.shape={env.action_space.shape}")

  action_dim = env.action_space.shape[-1]
  actions = torch.zeros(args.num_envs, action_dim, device=args.device, dtype=torch.float32)

  try:
    env.reset()
    if args.action_noise > 0.0:
      print(f"Stepping {args.steps} with action noise std={args.action_noise}")
    else:
      print(f"Stepping {args.steps} with zero actions")

    t1 = time.perf_counter()
    for _ in range(args.steps):
      if args.action_noise > 0.0:
        actions.normal_(0.0, args.action_noise)
      else:
        actions.zero_()
      env.step(actions)
    torch.cuda.synchronize() if args.device.startswith("cuda") else None
    t_step = time.perf_counter() - t1
    sps = args.steps / t_step
    print(f"Done: {args.steps} steps in {t_step:.3f}s  ({sps:.1f} env-steps/s, {sps * args.num_envs:.0f} world-steps/s)")
  finally:
    env.close()


if __name__ == "__main__":
  main()
