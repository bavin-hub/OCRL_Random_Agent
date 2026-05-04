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


from runners.train import Trainer
import json
import numpy as np

TASK_ID = "Mjlab-Velocity-Flat-Unitree-G1"
NUM_ENVS = 4096
NUM_STEPS = 24
MAX_ITERATIONS = 2600
SAVE_INTERVAL = 100


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


    ############## world model inits ##############
    # load config
    try:
        with open('config.json', 'r') as file:
            config = json.load(file)
            print('config file loaded successfully')
        file.close()
    except:
        raise FileNotFoundError("'config.json' file not found in the current dir")

    trainer = Trainer(config=config)
    



    ############## training loop ##############
    print(f"[INFO] task={TASK_ID} device={device} num_envs={NUM_ENVS} obs_dim={obs_dim} actions={num_actions}")

    for it in range(MAX_ITERATIONS):
        with torch.no_grad():
            for _ in range(NUM_STEPS):
                robot = env.unwrapped.scene["robot"]
                base_lin_vel = robot.data.root_link_lin_vel_w.to(device)      # (num_envs, 3) world frame
                base_ang_vel = robot.data.root_link_ang_vel_w.to(device)      # (num_envs, 3) world frame
                projected_gravity = robot.data.projected_gravity_b.to(device)  # (num_envs, 3) base frame
                joint_pos = robot.data.joint_pos.to(device)                   # (num_envs, num_joints)
                joint_vel = robot.data.joint_vel.to(device)   
                tau = robot.data.actuator_force.to(device)

                # single_env_proj_gravity = torch.linalg.vector_norm(projected_gravity[:100, :], dim=-1)
                # print(single_env_proj_gravity, "\n\n")

                robot_state = torch.concat([base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, tau], dim=-1)
                # print(from_env.shape)
                # print("directly from robot")
                # print(from_env[0, :].detach().cpu().numpy())
                
                actions = algo.act(obs_td)
                next_obs, rewards, dones, extras = env.step(actions.to(env.device))
                check_nan(next_obs, rewards, dones)

                next_obs = next_obs.to(device)
                rewards = rewards.to(device)
                dones = dones.to(device)
                next_obs_td = to_obs_tensordict(next_obs, device)

                algo.process_env_step(next_obs_td, rewards, dones, extras)

                # update obs, actions, dones to rwm replay_buffer that is inside our world model code
                # print(obs_td["policy"][:, :67].shape)
                # print("from mjlab")
                # print(obs_td["policy"][0, :67].detach().cpu().numpy())
                # np.set_printoptions(suppress=True, precision=4)
                # print("\n\n")



                trainer.insert_into_replay_buffer(robot_state, 
                                                  tau, 
                                                  actions, 
                                                  dones)

                obs_td = next_obs_td


        # train our world model from the buffer data
        trainer.on_the_fly_update(it)

        # if warmup done:
            # if not init imagination:
                # init_imagination
            # imagine()
            # algo.update()


        algo.compute_returns(obs_td)

        loss_dict = algo.update()
        print(f"Iter {it}: {loss_dict}")
        if it % SAVE_INTERVAL == 0:
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


    
    