from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from tensordict import TensorDict

import mjlab.tasks 
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from rsl_rl.models import MLPModel


def to_obs_tensordict(obs: torch.Tensor | TensorDict, device: str) -> TensorDict:
    if isinstance(obs, TensorDict):
        if "policy" in obs.keys():
            policy_obs = obs["policy"]
        else:
            first_key = next(iter(obs.keys()))
            policy_obs = obs[first_key]
    else:
        policy_obs = obs

    if policy_obs.dim() == 1:
        policy_obs = policy_obs.unsqueeze(-1)
    elif policy_obs.dim() > 2:
        policy_obs = policy_obs.flatten(start_dim=1)

    policy_obs = policy_obs.to(device)
    return TensorDict({"policy": policy_obs}, batch_size=[policy_obs.shape[0]], device=device)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MJLab with a saved actor checkpoint.")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("saved_models") / "model_0.pt",
        help="Path to .pt from test1.save_model (contains actor_state_dict).",
    )
    p.add_argument("--task-id", type=str, default="Mjlab-Velocity-Flat-Unitree-G1")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--steps", type=int, default=1000, help="Used when --viewer none.")
    p.add_argument("--device", type=str, default=None, help="Policy device; default auto.")
    p.add_argument("--clip-actions", type=float, default=100.0)
    p.add_argument(
        "--render-mode",
        type=str,
        default="none",
        help="Env render_mode: 'none' (default) or 'rgb_array' for offscreen env.render(). "
        "Does not open a GUI; use --viewer for that.",
    )
    p.add_argument(
        "--viewer",
        type=str,
        choices=("none", "auto", "native", "viser"),
        default="auto",
        help="Interactive sim: native MuJoCo window (needs DISPLAY) or viser browser. "
        "'none' runs headless for --steps. Default 'auto' picks native if a display is set.",
    )
    return p.parse_args()


def _render_mode_arg(value: str) -> str | None:
    if value.lower() in ("none", "null", ""):
        return None
    return value


def _resolve_viewer(choice: str) -> str:
    if choice == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        return "native" if has_display else "viser"
    return choice


class ActorPolicy:

    def __init__(self, actor: MLPModel, policy_device: str, action_device: torch.device | str) -> None:
        self.actor = actor
        self.policy_device = policy_device
        self.action_device = action_device

    def __call__(self, obs: torch.Tensor | TensorDict) -> torch.Tensor:
        obs_td = to_obs_tensordict(obs, self.policy_device)
        self.actor.eval()
        with torch.inference_mode():
            actions = self.actor(obs_td, stochastic_output=False)
        return actions.to(self.action_device)


def _build_env_and_actor(args: argparse.Namespace, device: str, play_cfg: bool):
    ckpt_path = args.checkpoint.resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    env_cfg = load_env_cfg(args.task_id, play=play_cfg)
    env_cfg.scene.num_envs = args.num_envs
    render_mode = _render_mode_arg(args.render_mode)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
    env = RslRlVecEnvWrapper(env, clip_actions=args.clip_actions)

    obs = env.get_observations().to(device)
    obs_td = to_obs_tensordict(obs, device)
    num_actions = env.num_actions
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    actor = MLPModel(
        obs_td,
        obs_groups,
        "actor",
        num_actions,
        hidden_dims=[256, 256, 256],
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ).to(device)

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "actor_state_dict" not in payload:
        raise KeyError(f"Checkpoint missing 'actor_state_dict': keys={list(payload.keys())}")
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    actor.eval()

    policy = ActorPolicy(actor, policy_device=device, action_device=env.device)
    return env, actor, policy, num_actions, ckpt_path, render_mode


def main() -> None:
    args = parse_args()
    configure_torch_backends()

    if args.device is not None:
        device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    use_viewer = args.viewer != "none"
    env, _actor, policy, num_actions, ckpt_path, render_mode = _build_env_and_actor(
        args, device, play_cfg=use_viewer
    )

    print(
        f"[INFO] task={args.task_id} checkpoint={ckpt_path} device={device} "
        f"viewer={args.viewer} render_mode={render_mode!r} num_envs={args.num_envs} actions={num_actions}"
    )

    try:
        if use_viewer:
            resolved = _resolve_viewer(args.viewer)
            print(
                f"[INFO] Launching interactive viewer ({resolved}). "
                f"Close the window or stop the server to exit. "
                f"If native fails, try: --viewer viser"
            )
            if resolved == "native":
                NativeMujocoViewer(env, policy).run()
            elif resolved == "viser":
                ViserPlayViewer(env, policy).run()
            else:
                raise RuntimeError(f"Unsupported viewer: {resolved}")
        else:
            obs = env.get_observations().to(device)
            obs_td = to_obs_tensordict(obs, device)
            ep_return = torch.zeros(args.num_envs, device=device)
            ep_len = torch.zeros(args.num_envs, device=device)

            with torch.inference_mode():
                for step in range(args.steps):
                    actions = policy(obs_td)
                    next_obs, rewards, dones, _extras = env.step(actions)
                    next_obs = next_obs.to(device)
                    rewards = rewards.to(device).view(-1)
                    dones = dones.to(device).view(-1).float()

                    ep_return += rewards * (1.0 - dones)
                    ep_len += 1.0
                    done_mask = dones > 0.5
                    if done_mask.any():
                        for i in done_mask.nonzero(as_tuple=False).view(-1).tolist():
                            print(
                                f"[episode] env={i} return={ep_return[i].item():.3f} "
                                f"len={int(ep_len[i].item())} at_step={step}"
                            )
                        ep_return = ep_return * (1.0 - dones)
                        ep_len = ep_len * (1.0 - dones)

                    obs_td = to_obs_tensordict(next_obs, device)

                    if (step + 1) % max(1, args.steps // 10) == 0:
                        print(
                            f"[progress] step={step + 1}/{args.steps} "
                            f"mean_reward_step={rewards.mean().item():.4f}"
                        )
    finally:
        env.close()
        print("[INFO] eval finished.")


if __name__ == "__main__":
    main()
