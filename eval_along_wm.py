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
import json
import numpy as np
from utils import CreateWorlModelInstance, LoadModel, z_norm, plot_graphs, minmax_norm_state_action_pair


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


def scale_policy_actions(policy_actions, scale, offset):
        return (policy_actions*scale) + offset


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



    # load config
    try:
        with open('config.json', 'r') as file:
            config = json.load(file)
            print('config file loaded successfully')
        file.close()
    except:
        raise FileNotFoundError("'config.json' file not found in the current dir")
    
    # wm configs
    scale = torch.tensor([config.get("scale")]).to(config.get("device"))
    offset = torch.tensor([config.get("offset")]).to(config.get("device"))

    mean_state_action = config.get("mean_state_action")
    std_state_action = config.get("std_state_action")
    state_mean = torch.tensor(mean_state_action[:96]).to(config.get("device"))
    state_std = torch.tensor(std_state_action[:96]).to(config.get("device"))
    action_mean = torch.tensor(mean_state_action[96:]).to(config.get("device"))
    action_std = torch.tensor(std_state_action[96:]).to(config.get("device"))
    jmin = torch.tensor([-2.5306999683380127, -0.5235999822616577, -2.7576000690460205, -0.08726699650287628, -0.8726699948310852, -0.26179999113082886, -2.5306999683380127, -2.967099905014038, -2.7576000690460205, -0.08726699650287628, -0.8726699948310852, -0.26179999113082886, -2.618000030517578, -0.5199999809265137, -0.5199999809265137, -3.089200019836426, -1.5881999731063843, -2.618000030517578, -1.0471999645233154, -1.9722199440002441, -1.6144299507141113, -1.6144299507141113, -3.089200019836426, -2.251499891281128, -2.618000030517578, -1.0471999645233154, -1.9722199440002441, -1.6144299507141113, -1.6144299507141113]).to(config.get("device"))
    jmax = torch.tensor([2.8798000812530518, 2.967099905014038, 2.7576000690460205, 2.8798000812530518, 0.5235999822616577, 0.26179999113082886, 2.8798000812530518, 0.5235999822616577, 2.7576000690460205, 2.8798000812530518, 0.5235999822616577, 0.26179999113082886, 2.618000030517578, 0.5199999809265137, 0.5199999809265137, 2.6703999042510986, 2.251499891281128, 2.618000030517578, 2.094399929046631, 1.9722199440002441, 1.6144299507141113, 1.6144299507141113, 2.6703999042510986, 1.5881999731063843, 2.618000030517578, 2.094399929046631, 1.9722199440002441, 1.6144299507141113, 1.6144299507141113]).to(config.get("device"))
    tau_min = torch.tensor([-25.0, -25.0, -25.0, -25.0, -25.0, -25.0, -25.0, -25.0, -25.0, -25.0, -88.0, -88.0, -88.0, -88.0, -88.0, -139.0, -139.0, -139.0, -139.0, -5.0, -5.0, -5.0, -5.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0]).to(config.get("device"))
    tau_max = torch.tensor([25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 88.0, 88.0, 88.0, 88.0, 88.0, 139.0, 139.0, 139.0, 139.0, 5.0, 5.0, 5.0, 5.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]).to(config.get("device"))


    print(scale)
    print(offset)
    print(state_mean)
    print(state_std)
    print(action_mean)
    print(action_std)
    print("\n\n\n")


    # load world model
    world_model = CreateWorlModelInstance(config)
    model_name = "wm-itr_2500.pth"
    model_dir_name = "wm_gru_2026-05-08_21:58:14"
    world_model = LoadModel(world_model, model_name, model_dir_name)
    print("successfully loaded world model")

    # init state action buffers to store (TODO)
    wm_transition_rows: list[torch.Tensor] = []

    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    use_viewer = args.viewer != "none"
    env, _actor, policy, num_actions, ckpt_path, render_mode = _build_env_and_actor(
        args, device, play_cfg=use_viewer
    )

    print(
        f"[INFO] task={args.task_id} checkpoint={ckpt_path} device={device} "
        f"viewer={args.viewer} render_mode={render_mode!r} num_envs={args.num_envs} actions={num_actions}"
    )

    IMAGINATION = False

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
            last_actions = torch.zeros(args.num_envs, 29).to(config.get("device"))
            with torch.inference_mode():
                for step in range(args.steps):                
                    ############### curr robot state ############### 
                    robot = env.unwrapped.scene["robot"]
                    base_lin_vel = robot.data.root_link_lin_vel_w.to(device)      # (num_envs, 3) world frame
                    base_ang_vel = robot.data.root_link_ang_vel_w.to(device)      # (num_envs, 3) world frame
                    projected_gravity = robot.data.projected_gravity_b.to(device)  # (num_envs, 3) base frame
                    joint_pos = robot.data.joint_pos.to(device)                   # (num_envs, num_joints)
                    joint_vel = robot.data.joint_vel.to(device)   
                    tau = robot.data.actuator_force.to(device)
                    robot_state = torch.concat([base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, tau], dim=-1)
                    ############### curr robot state end ###############

                    ############ norm the curr robot state ############
                    # last_dummy_actions = obs_td["policy"][:, 67:96]
                    cmd_vel = obs_td["policy"][:, -3:]
                    normed_robot_state, _ = z_norm(robot_state, last_actions, 
                                                state_mean, state_std,
                                                action_mean, action_std)
                    policy_input_obs = torch.concat([normed_robot_state[:, :67],
                                                    last_actions,
                                                    cmd_vel], dim=-1)
                    obs_td["policy"] = policy_input_obs
                    ############ norm the curr robot state end ############

                    # if IMAGINATION:
                    #     # modify the obs accordingly
                    #     mjlab_obs = obs_td["policy"]
                    #     last_action_dummy = mjlab_obs[:, 67:96]
                    #     cmd_vels = mjlab_obs[:, 96:]
                    #     robot_state_normed, _ = z_norm(robot_state, last_action_dummy, 
                    #                                     state_mean, state_std,
                    #                                     action_mean, action_std)
                        
                    #     wm_obs = torch.concat([robot_state_normed[:, :67],
                    #                            last_action_dummy,
                    #                            cmd_vels], dim=-1)
                    #     obs_td["policy"] = wm_obs
                        

                   

                    actions = policy(obs_td)
                    last_actions = scale_policy_actions(actions, scale, offset)

                    next_obs, rewards, dones, _extras = env.step(actions)
                    next_obs = next_obs.to(device)
                    rewards = rewards.to(device).view(-1)
                    dones = dones.to(device).view(-1).float()

                    ############### next robot state ############### 
                    next_robot = env.unwrapped.scene["robot"]
                    base_lin_vel = next_robot.data.root_link_lin_vel_w.to(device)      # (num_envs, 3) world frame
                    base_ang_vel = next_robot.data.root_link_ang_vel_w.to(device)      # (num_envs, 3) world frame
                    projected_gravity = next_robot.data.projected_gravity_b.to(device)  # (num_envs, 3) base frame
                    joint_pos = next_robot.data.joint_pos.to(device)                   # (num_envs, num_joints)
                    joint_vel = next_robot.data.joint_vel.to(device)   
                    tau = next_robot.data.actuator_force.to(device)
                    next_robot_state = torch.concat([base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, tau], dim=-1)
                    ############### next robot state end ###############


                    ############ norm the next robot state ############
                    # last_dummy_actions = next_obs[:, 67:96]
                    cmd_vel = next_obs["actor"][:, -3:]
                    normed_robot_state, _ = z_norm(next_robot_state, last_actions, 
                                                state_mean, state_std,
                                                action_mean, action_std)
                    policy_input_obs = torch.concat([normed_robot_state[:, :67],
                                                    last_actions,
                                                    cmd_vel], dim=-1)
                    next_obs["actor"] = policy_input_obs
                    ############ norm the next robot state end ############
                    

                    ############### pre-processing required for world model ###############
                    policy_obs = obs_td["policy"]
                    obs_vec = torch.cat((policy_obs[..., :67], tau), dim=-1)

                    # scale actions using the above function (TODO)
                    scaled_act = scale_policy_actions(
                        actions, scale.to(device), offset.to(device)
                    )
                    # normalize state and scaled actions using imported z_norm function (TODO)
                    obs_norm, action_norm = z_norm(
                        robot_state,
                        scaled_act,
                        state_mean.to(device),
                        state_std.to(device),
                        action_mean.to(device),
                        action_std.to(device),
                    )
                    
                    # state_action_pair = torch.concat([robot_state, scaled_act], dim=-1)
                    # obs_norm, action_norm = minmax_norm_state_action_pair(state_action_pair, 
                    #                                                       jmin,
                    #                                                       jmax,
                    #                                                       tau_min,
                    #                                                       tau_max)
                    # print(obs_norm.shape)
                    wm_transition_rows.append(
                        torch.cat([obs_norm[0], action_norm[0]], dim=-1).detach().cpu()
                    )
                    ############### end of preprocessing ###############

                    
                    ep_return += rewards * (1.0 - dones)
                    ep_len += 1.0
                    done_mask = dones > 0.5
                    if done_mask.any():
                        print("i think i am done")
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

            # sample M + N_pred (defined in config) transitions (TODO)
            # iterate through M + N_pred steps (refer to /runners/eval.py) (TODO)
            # store the world model predicted transitions (TODO)
            # plot ground and predicted states (including though the wm is not going to predict till M you can use ground truth states) (TODO)
            # show the graph (don't have to save the graph) (TODO)
            M_wm = config["world_model_training_params"]["M"]
            N_pred = config["world_model_training_params"]["N_pred"]
            need_steps = M_wm + N_pred
            action_dims = config["robot_params"]["action_dims"]
            max_steps = M_wm + N_pred
            JOINT_PICK = [i for i in range(29)]

            if len(wm_transition_rows) >= need_steps:
                window = torch.stack(wm_transition_rows[-need_steps:], dim=0)
                single_trajectory = window.unsqueeze(0).float()
                wm_dev = next(world_model.parameters()).device
                world_model.eval()
                ht = torch.zeros(
                    (
                        config["world_model_arch_params"]["num_gru_layers"],
                        single_trajectory.shape[0],
                        config["world_model_arch_params"]["gru_hidden_dim"],
                    ),
                    device=wm_dev,
                )
                x = single_trajectory.to(wm_dev)

                n_joints = config["robot_params"]["action_dims"]
                state_dims = config["robot_params"]["state_dims"]
                # [base_lin_vel(3), base_ang_vel(3), proj_gravity(3), q(n), qd(n), tau(n)]
                _expected = 9 + 3 * n_joints
                if state_dims != _expected:
                    raise ValueError(
                        f"state_dims={state_dims} does not match 9+3*n_joints={_expected} (n_joints={n_joints})"
                    )
                off_qd = 9 + n_joints
                off_tau = 9 + 2 * n_joints

                lin_vel_x_pred: list[float] = []
                lin_vel_y_pred: list[float] = []
                lin_vel_z_pred: list[float] = []
                ang_vel_x_pred: list[float] = []
                ang_vel_y_pred: list[float] = []
                ang_vel_z_pred: list[float] = []
                grav_x_pred: list[float] = []
                grav_y_pred: list[float] = []
                grav_z_pred: list[float] = []
                pred_joints = {j: [] for j in JOINT_PICK}
                pred_joint_vel = {j: [] for j in JOINT_PICK}
                pred_joint_tau = {j: [] for j in JOINT_PICK}

                # print("till here")
                # print(x.shape)
                # print(x)
                # print("\n\n")

                with torch.inference_mode():
                    for t in range(max_steps - 1):
                        if t < M_wm - 1:
                            single_transition = torch.squeeze(x[:, t, :], dim=0).clone().detach()
                            single_transition = single_transition[:96].to("cpu").numpy().tolist()
                            ht = world_model.forward(
                                torch.unsqueeze(x[:, t, :], dim=1), ht, predict=False
                            )
                        else:
                            if t == M_wm - 1:
                                x_prev = torch.unsqueeze(x[:, t, :96], dim=1)
                                st_next_pred, ht = world_model.forward(
                                    torch.unsqueeze(x[:, t, :], dim=1),
                                    ht,
                                    predict=True,
                                    sample=True,
                                    x_prev=x_prev,
                                )
                            else:
                                st_next_pred, ht = world_model.forward(
                                    torch.cat(
                                        (
                                            st_next_pred,
                                            torch.unsqueeze(x[:, t, -action_dims:], dim=1),
                                        ),
                                        dim=2,
                                    ),
                                    ht,
                                    predict=True,
                                    sample=True,
                                    x_prev=st_next_pred,
                                )
                            single_transition = (
                                torch.squeeze(st_next_pred[-1], dim=0).clone().detach().to("cpu").numpy().tolist()
                            )

                        state_vec = single_transition
                        pred_velocities = state_vec[:6]
                        lin_vel_x_pred.append(pred_velocities[0])
                        lin_vel_y_pred.append(pred_velocities[1])
                        lin_vel_z_pred.append(pred_velocities[2])
                        ang_vel_x_pred.append(pred_velocities[3])
                        ang_vel_y_pred.append(pred_velocities[4])
                        ang_vel_z_pred.append(pred_velocities[5])
                        grav_x_pred.append(state_vec[6])
                        grav_y_pred.append(state_vec[7])
                        grav_z_pred.append(state_vec[8])
                        for j in JOINT_PICK:
                            pred_joints[j].append(state_vec[9 + j])
                            pred_joint_vel[j].append(state_vec[off_qd + j])
                            pred_joint_tau[j].append(state_vec[off_tau + j])

                lin_vel_x: list[float] = []
                lin_vel_y: list[float] = []
                lin_vel_z: list[float] = []
                ang_vel_x: list[float] = []
                ang_vel_y: list[float] = []
                ang_vel_z: list[float] = []
                grav_x: list[float] = []
                grav_y: list[float] = []
                grav_z: list[float] = []
                true_joints = {j: [] for j in JOINT_PICK}
                true_joint_vel = {j: [] for j in JOINT_PICK}
                true_joint_tau = {j: [] for j in JOINT_PICK}
                true_traj = np.squeeze(single_trajectory.detach().cpu().numpy())
                for i in range(max_steps - 1):
                    transition = true_traj[i, :]
                    state = transition[:state_dims].tolist()
                    velocities = state[:6]
                    print(velocities, "\n\n")
                    lin_vel_x.append(velocities[0])
                    lin_vel_y.append(velocities[1])
                    lin_vel_z.append(velocities[2])
                    ang_vel_x.append(velocities[3])
                    ang_vel_y.append(velocities[4])
                    ang_vel_z.append(velocities[5])
                    grav_x.append(state[6])
                    grav_y.append(state[7])
                    grav_z.append(state[8])
                    for j in JOINT_PICK:
                        true_joints[j].append(state[9 + j])
                        true_joint_vel[j].append(state[off_qd + j])
                        true_joint_tau[j].append(state[off_tau + j])

                lin_vel_true = [lin_vel_x, lin_vel_y, lin_vel_z]
                lin_vel_preds = [lin_vel_x_pred, lin_vel_y_pred, lin_vel_z_pred]
                ang_vel_true = [ang_vel_x, ang_vel_y, ang_vel_z]
                ang_vel_preds = [ang_vel_x_pred, ang_vel_y_pred, ang_vel_z_pred]

                proj_grav_true = [grav_x, grav_y, grav_z]
                proj_grav_pred = [grav_x_pred, grav_y_pred, grav_z_pred]

                plot_graphs(
                    lin_vel_true=lin_vel_true,
                    lin_vel_preds=lin_vel_preds,
                    ang_vel_true=ang_vel_true,
                    ang_vel_preds=ang_vel_preds,
                    true_joints=true_joints,
                    pred_joints=pred_joints,
                    M=M_wm,
                    model_dir_name=model_dir_name,
                    model_name=model_name,
                    proj_grav_true=proj_grav_true,
                    proj_grav_pred=proj_grav_pred,
                    true_joint_vel=true_joint_vel,
                    pred_joint_vel=pred_joint_vel,
                    true_joint_tau=true_joint_tau,
                    pred_joint_tau=pred_joint_tau,
                )
            else:
                print(
                    f"[WARN] world model plot skipped: need {need_steps} transitions, "
                    f"collected {len(wm_transition_rows)}"
                )

    finally:
        env.close()
        print("[INFO] eval finished.")


if __name__ == "__main__":
    main()



# 
'''
Exps:

1. vis the code and see if there is any mistake
2. change the znorm to min-max norm
3. reduce learning rate, increase num mini batches, increase num epochs
4. more iterations
5. go through their code base to clear out some doubts
6. try raw mse once (no dist. shit)
7. tweak arch hyper params
8. plot all graphs
9. try setting up their code
10. apply tanh bounds
'''