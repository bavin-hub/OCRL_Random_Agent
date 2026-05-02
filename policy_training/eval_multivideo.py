"""Evaluate a trained RL policy on MuJoCo and save multiple videos.

Usage:
  python policy_training/eval_multivideo.py Unitree-G1-Flat-MJ \
      --checkpoint logs/rsl_rl/g1_velocity/2026-04-13_00-53-31/model_10000.pt \
      --video --num-videos 3 --video-length 200
"""

import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class EvalConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  checkpoint: str = ""
  num_episodes: int = 50
  use_world_model: bool = False
  world_model_db_dir: str = "pretraining_rollouts/1000000_transitions"
  world_model_checkpoint: str = ""
  video: bool = False
  video_length: int = 200
  num_videos: int = 3
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "EvalConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return EvalConfig(env=env_cfg, agent=agent_cfg)


def run_eval(task_id: str, cfg: EvalConfig):
  configure_torch_backends()

  if not cfg.checkpoint:
    raise ValueError("--checkpoint is required")
  checkpoint_path = Path(cfg.checkpoint)
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  device = "cpu" if cuda_visible == "" else "cuda:0"

  cfg.env.seed = cfg.agent.seed

  print(f"[INFO] Evaluating: task={task_id}, device={device}")
  print(f"[INFO] Checkpoint: {checkpoint_path}")
  print(f"[INFO] Episodes: {cfg.num_episodes}")

  # Create environment.
  render_mode = "rgb_array" if (cfg.video and not cfg.use_world_model) else None

  if cfg.use_world_model:
    import json
    from utils import CreateWorlModelInstance
    with open("config.json", "r") as f:
      wm_cfg = json.load(f)
    wm_cfg["device"] = device
    world_model = CreateWorlModelInstance(wm_cfg)
    if cfg.world_model_checkpoint:
      print(f"[INFO] Loading World Model from {cfg.world_model_checkpoint}")
      world_model.load_state_dict(
        torch.load(cfg.world_model_checkpoint, map_location=device, weights_only=True)
      )
    from policy_training.world_model_env import WorldModelEnv
    env = WorldModelEnv(cfg=cfg.env, world_model=world_model, db_dir_name=cfg.world_model_db_dir, device=device)
    print("[INFO] Using World Model environment")
    if cfg.video:
      print("[WARN] Video recording not available with World Model (no renderer)")
  else:
    env = ManagerBasedRlEnv(cfg=cfg.env, device=device, render_mode=render_mode)
    print("[INFO] Using MuJoCo environment")

  if cfg.video and not cfg.use_world_model:
    video_dir = checkpoint_path.parent / "videos" / "eval_multivideo"
    print(f"[INFO] Recording {cfg.num_videos} videos to {video_dir}")
    
    # Calculate step triggers to record non-overlapping videos
    trigger_steps = [i * cfg.video_length for i in range(cfg.num_videos)]
    
    env = VideoRecorder(
      env,
      video_folder=video_dir,
      step_trigger=lambda step: step in trigger_steps,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  # Load policy.
  agent_cfg = asdict(cfg.agent)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, agent_cfg, device=device)
  runner.load(str(checkpoint_path), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  print(f"\n{'='*60}")
  print(f"  Running {cfg.num_episodes} evaluation episodes")
  print(f"{'='*60}\n")

  # Run episodes.
  all_rewards = []
  all_lengths = []

  env.reset()

  episode_rewards = torch.zeros(env.unwrapped.num_envs, device=device)
  episode_lengths = torch.zeros(env.unwrapped.num_envs, device=device, dtype=torch.long)
  completed = 0

  while completed < cfg.num_episodes:
    with torch.no_grad():
      obs = env.get_observations()
      actions = policy(obs)

    _, rewards, dones, extras = env.step(actions)

    episode_rewards += rewards
    episode_lengths += 1

    # Collect finished episodes.
    done_mask = dones.bool()
    if done_mask.any():
      done_ids = done_mask.nonzero(as_tuple=False).flatten()
      for idx in done_ids:
        if completed >= cfg.num_episodes:
          break
        ep_ret = episode_rewards[idx].item()
        ep_len = episode_lengths[idx].item()
        all_rewards.append(ep_ret)
        all_lengths.append(ep_len)
        completed += 1

        if completed % 10 == 0 or completed == cfg.num_episodes:
          print(f"  Episodes completed: {completed}/{cfg.num_episodes}")

      # Reset counters for finished envs.
      episode_rewards[done_ids] = 0.0
      episode_lengths[done_ids] = 0

  # Print results.
  rewards_t = torch.tensor(all_rewards)
  lengths_t = torch.tensor(all_lengths, dtype=torch.float32)

  print(f"\n{'='*60}")
  print(f"  Evaluation Results ({cfg.num_episodes} episodes)")
  print(f"{'='*60}")
  print(f"  Mean Return:      {rewards_t.mean().item():>10.2f} +/- {rewards_t.std().item():.2f}")
  print(f"  Median Return:    {rewards_t.median().item():>10.2f}")
  print(f"  Min / Max Return: {rewards_t.min().item():>10.2f} / {rewards_t.max().item():.2f}")
  print(f"  Mean Ep Length:   {lengths_t.mean().item():>10.1f} +/- {lengths_t.std().item():.1f}")
  print(f"  Min / Max Length: {lengths_t.min().item():>10.0f} / {lengths_t.max().item():.0f}")
  print(f"{'='*60}\n")

  env.close()


def main():
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  try:
      import policy_training.local_env  # noqa: F401
  except ImportError:
      pass

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    EvalConfig,
    args=remaining_args,
    default=EvalConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  run_eval(task_id=chosen_task, cfg=args)


if __name__ == "__main__":
  main()
