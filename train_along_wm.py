from __future__ import annotations

import torch
from pathlib import Path
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import check_nan

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

from utils import z_norm, load_ppo_policy_from_checkpoint
from runners.train import Trainer
import json
import numpy as np

TASK_ID = "Mjlab-Velocity-Flat-Unitree-G1"
NUM_ENVS = 4096
NUM_STEPS = 24
MAX_ITERATIONS = 2600
SAVE_INTERVAL = 10
IMAGINATION = True
# EMA decay for per-iteration mean reward (closer to 1.0 = smoother / slower to move).
REWARD_EMA_DECAY = 0.995
LOAD_WORLD_MODEL = True
LOAD_POLICY_CHECKPOINT = True
POLICY_CHECKPOINT_PATH = "saved_models_copy/model_2500.pt"


def format_episode_stats(
    label: str,
    finished_returns: list[torch.Tensor],
    finished_lengths: list[torch.Tensor],
) -> str:
    """Same reporting shape as ``train_custom_reward.py`` (mean/min/max return, mean ep length, count)."""
    if finished_returns:
        all_r = torch.cat(finished_returns)
        all_l = torch.cat(finished_lengths)
        return (
            f"{label} ep_return mean={all_r.mean().item():.2f} "
            f"min={all_r.min().item():.2f} max={all_r.max().item():.2f} "
            f"ep_length mean={all_l.float().mean().item():.1f} count={all_r.numel()}"
        )
    return f"{label}: no episodes finished this iter"


def to_obs_tensordict(obs: torch.Tensor | TensorDict, device: str) -> TensorDict:
    """Normalize observations into TensorDict with 2D 'policy' features."""
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


def save_model(algo: PPO, iteration: int, save_dir: str = "saved_models") -> Path:
    path = Path.cwd() / save_dir
    path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = path / f"model_{iteration}.pt"
    payload = algo.save()
    payload["iter"] = iteration
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def main() -> None:
    ############## rsl_rl inits ##############
    configure_torch_backends()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    env_cfg = load_env_cfg(TASK_ID)
    env_cfg.scene.num_envs = NUM_ENVS
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=100.0)

    obs = env.get_observations().to(device)
    obs_td = to_obs_tensordict(obs, device)

    obs_dim = obs_td["policy"].shape[-1]
    print(obs_dim)
    print("\n\n\n\n\n")
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

    critic = MLPModel(
        obs_td,
        obs_groups,
        "critic",
        1,
        hidden_dims=[256, 256, 256],
        activation="elu",
        obs_normalization=True,
    ).to(device)

    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs_td, [num_actions], device=device)
    algo = PPO(
        actor,
        critic,
        storage,
        clip_param=0.2,
        num_learning_epochs=5,
        num_mini_batches=4,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=3e-4,
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
        device=device,
    )
    algo.train_mode()

    if LOAD_POLICY_CHECKPOINT:
        saved_iter = load_ppo_policy_from_checkpoint(
            algo,
            POLICY_CHECKPOINT_PATH,
            load_optimizer=True,
            strict=True,
        )
        print(
            f"[INFO] loaded PPO policy from {POLICY_CHECKPOINT_PATH}"
            + (f" (checkpoint iter={saved_iter})" if saved_iter is not None else "")
        )

    ############## world model inits ##############
    # load config
    try:
        with open('config.json', 'r') as file:
            config = json.load(file)
            print('config file loaded successfully')
        file.close()
    except:
        raise FileNotFoundError("'config.json' file not found in the current dir")

    world_model = Trainer(config=config)
    if LOAD_WORLD_MODEL:
        ckpt_name = "wm-itr-ckpt-epoch_2500.pth"
        ckpt_dir = "wm_gru_2026-05-08_21:58:14"
        world_model.load_model(ckpt_name, ckpt_dir)
    
    WARM_START_PERIOD = 10
    WARM_START_CTR = 0
    ema_real_reward: float | None = None
    ema_imag_reward: float | None = None

    real_ep_return = torch.zeros(NUM_ENVS, device=device)
    real_ep_length = torch.zeros(NUM_ENVS, device=device, dtype=torch.long)
    finished_real_returns: list[torch.Tensor] = []
    finished_real_lengths: list[torch.Tensor] = []
    max_ep_len = int(env.unwrapped.max_episode_length)

    imag_ep_return = torch.zeros(NUM_ENVS, device=device)
    imag_ep_length = torch.zeros(NUM_ENVS, device=device, dtype=torch.long)
    finished_imag_returns: list[torch.Tensor] = []
    finished_imag_lengths: list[torch.Tensor] = []

    ############## training loop ##############
    print(f"[INFO] task={TASK_ID} device={device} num_envs={NUM_ENVS} obs_dim={obs_dim} actions={num_actions}")

    mean_state_action = config.get("mean_state_action")
    std_state_action = config.get("std_state_action")
    state_mean = torch.tensor(mean_state_action[:96]).to(config.get("device"))
    state_std = torch.tensor(std_state_action[:96]).to(config.get("device"))
    action_mean = torch.tensor(mean_state_action[96:]).to(config.get("device"))
    action_std = torch.tensor(std_state_action[96:]).to(config.get("device"))

    # storing cmd vels
    cmd_vels = []

    for it in range(MAX_ITERATIONS):
        real_reward_means: list[torch.Tensor] = []

        with torch.no_grad():
            last_actions = torch.zeros(NUM_ENVS, 29).to(config.get("device"))
            for _ in range(NUM_STEPS):

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

                cmd_vels.append(cmd_vel)
                # print("\n\n")
                
                actions = algo.act(obs_td)
                prev_actions = last_actions
                last_actions = world_model.scale_policy_actions(actions)
                next_obs, rewards, dones, extras = env.step(actions.to(env.device))
            
       

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
                normed_next_robot_state, _ = z_norm(next_robot_state, last_actions, 
                                               state_mean, state_std,
                                               action_mean, action_std)
                next_policy_input_obs = torch.concat([normed_next_robot_state[:, :67],
                                                 last_actions,
                                                 cmd_vel], dim=-1)
                next_obs["actor"] = next_policy_input_obs
                ############ norm the next robot state end ############

                ############ custom step ############
                real_ep_length += 1
                rewards, dones, extras = world_model.custom_step(cmd_vel, normed_robot_state, normed_next_robot_state, prev_actions, last_actions, real_ep_length, max_ep_len)
                ############ custom step end ############

                check_nan(next_obs, rewards, dones)

                next_obs = next_obs.to(device)
                rewards = rewards.to(device)
                dones = dones.to(device)
                real_reward_means.append(rewards.mean().detach())

                r_vec = rewards.float().reshape(-1)
                real_ep_return += r_vec
                done_mask_real = dones.reshape(-1).to(dtype=torch.bool, device=device)
                if done_mask_real.any():
                    finished_real_returns.append(real_ep_return[done_mask_real].clone())
                    finished_real_lengths.append(real_ep_length[done_mask_real].clone())
                    real_ep_return[done_mask_real] = 0.0
                    real_ep_length[done_mask_real] = 0

                next_obs_td = to_obs_tensordict(next_obs, device)


                if not IMAGINATION:
                    algo.process_env_step(next_obs_td, rewards, dones, extras)



                world_model.insert_into_replay_buffer(robot_state, 
                                                      tau, 
                                                      actions, 
                                                      dones)

                obs_td = next_obs_td

        real_mean_this_iter = float(torch.stack(real_reward_means).mean())
        if ema_real_reward is None:
            ema_real_reward = real_mean_this_iter
        else:
            ema_real_reward = REWARD_EMA_DECAY * ema_real_reward + (1.0 - REWARD_EMA_DECAY) * real_mean_this_iter

        real_ep_stats = format_episode_stats("real", finished_real_returns, finished_real_lengths)
        finished_real_returns.clear()
        finished_real_lengths.clear()

        # train our world model from the buffer data
        # if not LOAD_WORLD_MODEL:
        world_model.on_the_fly_update(it)


        WARM_START_CTR += 1

        if IMAGINATION and WARM_START_CTR > WARM_START_PERIOD:
            # wm based imagination update
            idx = 0

            # last obs command vels 
            # last_cmd_vels = obs_td["policy"][:, -3:].detach()
            
            # Long-horizon prefixes for warm-start / partial resets (normalized replay samples).
            prefix_state_hist, prefix_action_hist = world_model.get_warm_start_buffer()

            max_ep_len = int(env.unwrapped.max_episode_length)

            imag_reward_means: list[torch.Tensor] = []
            with torch.inference_mode():

                state_hist, action_hist, st_next_pred, at = world_model.reset(
                    prefix_state_hist, prefix_action_hist
                )

                for _i in range(NUM_STEPS):

                    last_cmd_vels = cmd_vels[idx].detach()
                    idx += 1

                    imagined_obs, at_denormalized = world_model.preprocess_obs(
                        st_next_pred, at, last_cmd_vels
                    )

                    imagined_actions = algo.act(to_obs_tensordict(imagined_obs, device))

                    imag_ep_length += 1
                    (
                        state_hist,
                        action_hist,
                        st_next_pred,
                        at,
                        imagined_obs_next,
                        rewards,
                        dones,
                        extras,
                    ) = world_model.imagination_step(
                        st_next_pred,
                        imagined_actions,
                        at_denormalized,
                        last_cmd_vels,
                        imag_ep_length,
                        max_ep_len,
                        state_hist,
                    )
                    imag_reward_means.append(rewards.mean().detach())

                    rw_imag = rewards.float().reshape(-1)
                    imag_ep_return += rw_imag

                    done_mask = dones.view(-1).to(dtype=torch.bool, device=device)
                    reset_ids = torch.nonzero(done_mask, as_tuple=False).flatten()
                    if done_mask.any():
                        finished_imag_returns.append(imag_ep_return[done_mask].clone())
                        finished_imag_lengths.append(imag_ep_length[done_mask].clone())
                        imag_ep_return[done_mask] = 0.0
                        imag_ep_length[done_mask] = 0
                    if reset_ids.numel() > 0:
                        imagination_generator = world_model.replay_buffer.mini_batch_generator(
                            32,
                            1,
                            int(reset_ids.numel()),
                        )
                        batch = next(imagination_generator)
                        imagination_state_history, imagination_action_history = batch[0], batch[1]
                        prefix_state_hist[reset_ids] = imagination_state_history
                        prefix_action_hist[reset_ids] = imagination_action_history
                        state_hist, action_hist, st_next_pred, at = world_model.reset(
                            prefix_state_hist,
                            prefix_action_hist,
                            env_indices=reset_ids,
                        )

                    next_obs_td = to_obs_tensordict(imagined_obs_next, device)
                    algo.process_env_step(next_obs_td, rewards, dones, extras)

                    obs_td = next_obs_td
                

            imag_mean_this_iter = float(torch.stack(imag_reward_means).mean())
            if ema_imag_reward is None:
                ema_imag_reward = imag_mean_this_iter
            else:
                ema_imag_reward = REWARD_EMA_DECAY * ema_imag_reward + (1.0 - REWARD_EMA_DECAY) * imag_mean_this_iter

            imag_ep_stats = format_episode_stats("imag", finished_imag_returns, finished_imag_lengths)
            finished_imag_returns.clear()
            finished_imag_lengths.clear()

            algo.compute_returns(obs_td)
            loss_dict = algo.update()
            print(
                f"\nIter (imagination) {it}: {real_ep_stats} | {imag_ep_stats} | {loss_dict} | "
                f"real_r_ema={ema_real_reward:.4f} imag_r_ema={ema_imag_reward:.4f} "
                f"(iter means: real={real_mean_this_iter:.4f} imag={imag_mean_this_iter:.4f})"
            )
            
        elif not IMAGINATION:
            algo.compute_returns(obs_td)
            loss_dict = algo.update()
            print(
                f"\nIter (mujoco) {it}: {real_ep_stats} | {loss_dict} | real_r_ema={ema_real_reward:.4f} "
                f"(iter mean: real={real_mean_this_iter:.4f})"
            )
        else:
            print(
                f"\nIter (warm, WM data only) {it}: {real_ep_stats} | real_r_ema={ema_real_reward:.4f} "
                f"(iter mean: real={real_mean_this_iter:.4f})"
            )

        cmd_vels.clear()



       
        if (it % SAVE_INTERVAL == 0) and (WARM_START_CTR > WARM_START_PERIOD):
            ckpt = save_model(algo, it)
            print(f"[INFO] saved checkpoint: {ckpt}")

    final_ckpt = save_model(algo, MAX_ITERATIONS - 1)
    print(f"[INFO] saved final checkpoint: {final_ckpt}")
    env.close()


if __name__ == "__main__":
    main()



# wm.imagine(algo)
# inference_mode()
# for step in num_steps_per_env:
#     actions = algo.act(obs)
#     next_obs, reward, done, extras = custom_step() # will call wm integrator and compute reward, done
#     algo.process_env_step(next_obs, rewards, dones, extras)
#     obs = next_obs
# algo.compute_returns(obs)


    
    