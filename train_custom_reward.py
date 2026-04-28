"""
usage:
python train_custom_reward.py
"""





from __future__ import annotations

import torch
from pathlib import Path
from tensordict import TensorDict

import mjlab.tasks  
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.mdp import foot_air_time, foot_contact
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import check_nan


TASK_ID = "Mjlab-Velocity-Flat-Unitree-G1"
NUM_ENVS = 4096
NUM_STEPS = 24
MAX_ITERATIONS = 3000
SAVE_INTERVAL = 100


# G1 joint constants (mirrors policy_training/world_model_env.py).
G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

G1_DEFAULT_JOINT_POS = [
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    0.0, 0.0, 0.0,
    0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
    0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
]

G1_SOFT_JOINT_LIMITS_LO = [
    -2.260175, -0.349065, -2.48184, 0.061086, -0.802856, -0.23562,
    -2.260175, -2.792565, -2.48184, 0.061086, -0.802856, -0.23562,
    -2.3562, -0.468, -0.468,
    -2.80122, -1.396215, -2.3562, -0.89012, -1.774998, -1.452987, -1.452987,
    -2.80122, -2.059515, -2.3562, -0.89012, -1.774998, -1.452987, -1.452987,
]

G1_SOFT_JOINT_LIMITS_HI = [
    2.609275, 2.792565, 2.48184, 2.731447, 0.453786, 0.23562,
    2.609275, 0.349065, 2.48184, 2.731447, 0.453786, 0.23562,
    2.3562, 0.468, 0.468,
    2.38242, 2.059515, 2.3562, 1.93732, 1.774998, 1.452987, 1.452987,
    2.38242, 1.396215, 2.3562, 1.93732, 1.774998, 1.452987, 1.452987,
]

G1_STD_STANDING = [0.05] * 29

_WALKING_STD_MAP = {
    "hip_pitch": 0.5, "hip_roll": 0.15, "hip_yaw": 0.15,
    "knee": 0.5, "ankle_pitch": 0.15, "ankle_roll": 0.1,
    "waist_yaw": 0.15, "waist_roll": 0.1, "waist_pitch": 0.1,
    "shoulder_pitch": 0.15, "shoulder_roll": 0.1, "shoulder_yaw": 0.1,
    "elbow": 0.1, "wrist_roll": 0.1, "wrist_pitch": 0.1, "wrist_yaw": 0.1,
}

_RUNNING_STD_MAP = {
    "hip_pitch": 0.5, "hip_roll": 0.25, "hip_yaw": 0.25,
    "knee": 0.5, "ankle_pitch": 0.25, "ankle_roll": 0.1,
    "waist_yaw": 0.25, "waist_roll": 0.1, "waist_pitch": 0.1,
    "shoulder_pitch": 0.25, "shoulder_roll": 0.1, "shoulder_yaw": 0.1,
    "elbow": 0.1, "wrist_roll": 0.1, "wrist_pitch": 0.1, "wrist_yaw": 0.1,
}


def _resolve_std(joint_names, std_map):
    out = []
    for name in joint_names:
        key = name.replace("left_", "").replace("right_", "").replace("_joint", "")
        out.append(std_map.get(key, 0.1))
    return out


G1_STD_WALKING = _resolve_std(G1_JOINT_NAMES, _WALKING_STD_MAP)
G1_STD_RUNNING = _resolve_std(G1_JOINT_NAMES, _RUNNING_STD_MAP)


def setup_reward_constants(device):
    default_jp = torch.tensor(G1_DEFAULT_JOINT_POS, device=device, dtype=torch.float32)
    lo_abs = torch.tensor(G1_SOFT_JOINT_LIMITS_LO, device=device, dtype=torch.float32)
    hi_abs = torch.tensor(G1_SOFT_JOINT_LIMITS_HI, device=device, dtype=torch.float32)
    # Pre-shift limits into the relative joint-pos frame used by the actor obs
    # (obs gives q - q_default, so compare against limits in the same frame).
    return {
        "default_joint_pos": default_jp,
        "soft_limits_lo_rel": lo_abs - default_jp,
        "soft_limits_hi_rel": hi_abs - default_jp,
        "std_standing": torch.tensor(G1_STD_STANDING, device=device, dtype=torch.float32),
        "std_walking": torch.tensor(G1_STD_WALKING, device=device, dtype=torch.float32),
        "std_running": torch.tensor(G1_STD_RUNNING, device=device, dtype=torch.float32),
    }


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


def cmd_vels_from_obs(obs: torch.Tensor | TensorDict) -> torch.Tensor:
    if isinstance(obs, TensorDict):
        if "actor" in obs.keys():
            vec = obs["actor"]
        elif "policy" in obs.keys():
            vec = obs["policy"]
        else:
            vec = obs[next(iter(obs.keys()))]
    else:
        vec = obs

    if vec.dim() == 1:
        vec = vec.unsqueeze(0)
    elif vec.dim() > 2:
        vec = vec.flatten(start_dim=1)

    return vec[..., -3:]


def save_model(algo: PPO, iteration: int, save_dir: str = "saved_models") -> Path:
    path = Path.cwd() / save_dir
    path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = path / f"model_{iteration}.pt"
    payload = algo.save()
    payload["iter"] = iteration
    torch.save(payload, checkpoint_path)
    return checkpoint_path

def custom_reward(
    cmd_vels: torch.Tensor,
    obs_td: TensorDict,
    next_obs_td: TensorDict,
    last_actions: torch.Tensor,
    actions: torch.Tensor,
    tau: torch.Tensor,
    foot_air_time_t: torch.Tensor,
    foot_contact_t: torch.Tensor,
    consts: dict,
) -> torch.Tensor:

    w_v_xy = 1.0
    w_v_z = 0.0          # folded into r_v_xy exp (mirrors world_model_env)
    w_q_tau = 0.0        # non-kinematic; not in world_model_env
    w_a_dot = -0.05
    w_c = 0.0
    w_fc = 1.0
    w_omega_z = 1.0
    w_omega_xy = -0.05
    w_q_ddot = -2.5e-7
    w_fa = 0.0
    w_foot_contact = 0.0  # not in world_model_env
    w_g = -1.0
    w_q_d = -1.0
    w_stand_still = -1.0
    w_joint_pos_limits = -10.0
    w_pose = 1.0

    sigma_v_xy = 0.25
    sigma_omega_z = 0.25
    step_dt = 0.02

    # w_v_xy = 1.0
    # w_v_z = -2.0
    # w_q_tau = -2.5e-5
    # w_a_dot = -0.05
    # w_c = 0.0
    # w_fc = 1.0
    # w_omega_z = 0.5
    # w_omega_xy = -0.05
    # w_q_ddot = -2.5e-7
    # w_fa = 1.0
    # w_foot_contact = 0.1
    # w_g = -5.0
    # w_q_d = -1.0
    # w_stand_still = -0.1

    # sigma_v_xy = 0.25
    # sigma_omega_z = 0.25
    # step_dt = 0.02

    def sq_l2(x: torch.Tensor) -> torch.Tensor:
        return (x * x).mean(dim=-1)

    policy = obs_td["policy"]
    next_policy = next_obs_td["policy"]
    n = policy.shape[0]
    device = policy.device
    dtype = policy.dtype

    c_xy = cmd_vels[:, :2]
    cz = cmd_vels[:, 2:3]
    a_dim = actions.shape[-1]
    tail = policy.shape[1] - 12 - a_dim - 3
    j_dim = tail // 2

    v_xy = next_policy[:, 0:2]
    vz = next_policy[:, 2:3]
    omega_xy = next_policy[:, 3:5]
    omega_z = next_policy[:, 5:6]
    g_xy = next_policy[:, 6:8]
    q = policy[:, 9 : 9 + j_dim]
    q_next = next_policy[:, 9 : 9 + j_dim]
    q_vel = policy[:, 9 + j_dim : 9 + 2 * j_dim]
    q_vel_next = next_policy[:, 9 + j_dim : 9 + 2 * j_dim]
    q_ddot = (q_vel_next - q_vel) / step_dt

    tau = tau.to(device=device, dtype=dtype)
    ft = foot_air_time_t.to(device=device, dtype=dtype)
    fc = foot_contact_t.to(device=device, dtype=dtype)
    cu_f = torch.zeros(n, device=device, dtype=dtype)
    hfc_f = torch.zeros(n, device=device, dtype=dtype)

    cz_f = cz.reshape(n, -1).squeeze(-1)
    oz_f = omega_z.reshape(n, -1).squeeze(-1)
    vz_f = vz.reshape(n, -1).squeeze(-1)

    # Track lin vel: fold xy and z error inside the exp (world_model_env style (ripoff from mjlab vel tracking))
    d_xy = c_xy - v_xy
    xy_err_sum = (d_xy * d_xy).sum(dim=-1)
    z_err_sum = vz_f * vz_f
    r_v_xy = w_v_xy * torch.exp(-(xy_err_sum + z_err_sum) / 0.25)

    # Track ang vel_z (world_model_env: exp(-z_err / 0.5))
    d_z = cz_f - oz_f
    r_omega_z = w_omega_z * torch.exp(-(d_z * d_z) / 0.5)

    # # Lin vel z (vertical base velocity); not a separate v_x penalty
    # r_v_z = w_v_z * 0.5 * (vz_f * vz_f)

    # # Ang vel_xy (roll/pitch rates)
    # r_omega_xy = w_omega_xy * 0.5 * sq_l2(omega_xy)

    # # Joint / actuator torques
    # r_q_tau = w_q_tau * 0.5 * sq_l2(tau)

    # # Joint acceleration
    # r_q_ddot = w_q_ddot * 0.5 * sq_l2(q_ddot)

    # # Action rate
    # r_a_dot = w_a_dot * 0.5 * sq_l2(actions - last_actions)

    # Lin vel z (vertical base velocity); not a separate v_x penalty
    r_v_z = w_v_z * (vz_f * vz_f)

    # Ang vel_xy (roll/pitch rates) — sum to mirror world_model_env
    r_omega_xy = w_omega_xy * (omega_xy * omega_xy).sum(dim=-1)

    # Joint / actuator torques (non-kinematic; weight zeroed)
    r_q_tau = w_q_tau  * sq_l2(tau) #disabled
    # r_q_tau = torch.clamp(r_q_tau, min=-1.0)

    # Joint acceleration — sum to mirror world_model_env
    r_q_ddot = w_q_ddot * (q_ddot * q_ddot).sum(dim=-1)

    # Action rate sum 
    r_a_dot = w_a_dot  * ((actions - last_actions) ** 2).sum(dim=-1)

    # Feet air time (mjlab-style: in-range per foot, gated by command magnitude)
    air_min, air_max = 0.05, 0.5
    in_air_range = (ft > air_min) & (ft < air_max)
    r_fa_air = torch.sum(in_air_range.float(), dim=1)
    cmd_scale = (
        torch.norm(cmd_vels[:, :2], dim=1) + torch.abs(cmd_vels[:, 2]) > 0.5
    ).to(dtype=dtype)
    r_fa = w_fa * r_fa_air * cmd_scale

    # Foot contact: reward feet on ground (sum of binary contacts per env)
    r_foot_c = w_foot_contact * torch.sum(fc, dim=1)

    # Undesired contacts (disabled)
    r_c = w_c * cu_f

    # Flat orientation (projected gravity xy) — sum to mirror world_model_env
    r_g = w_g * (g_xy * g_xy).sum(dim=-1)

    # Foot clearance (not in actor obs without critic terms)
    r_fc = w_fc * hfc_f

    # # Joint deviation from default (actor uses joint_pos relative; post-step pose)
    r_q_d = w_q_d * q_next.abs().sum(dim=-1)

    # stand_still: sum((q - q_default)²) gated by total_speed <= 0.1.
    # q_next is already (q_abs - q_default), so squared error is q_next**2.
    total_cmd_speed = torch.norm(cmd_vels[:, :2], dim=1) + torch.abs(cmd_vels[:, 2])
    stand_still = (q_next * q_next).sum(dim=-1) * (total_cmd_speed <= 0.1).to(dtype=dtype)
    r_stand_still = w_stand_still * stand_still

    # joint_pos_limits (soft limit penalty). q is in joint_pos_rel frame
    # (q_abs - q_default), and consts["soft_limits_*_rel"] are pre-shifted
    # into the same frame at startup.
    lo_rel = consts["soft_limits_lo_rel"][:j_dim]
    hi_rel = consts["soft_limits_hi_rel"][:j_dim]
    below = -(q_next - lo_rel).clamp(max=0.0)
    above = (q_next - hi_rel).clamp(min=0.0)
    r_joint_pos_limits = w_joint_pos_limits * torch.sum(below + above, dim=1)

    # pose / variable_posture: speed-dependent posture tracking. Since q_next
    # is already (q_abs - q_default), the squared error is just q_next**2.
    total_speed = torch.norm(cmd_vels[:, :2], dim=1) + torch.abs(cmd_vels[:, 2])
    standing = (total_speed < 0.5).to(dtype=dtype).unsqueeze(1)
    walking = ((total_speed >= 0.5) & (total_speed < 1.5)).to(dtype=dtype).unsqueeze(1)
    running = (total_speed >= 1.5).to(dtype=dtype).unsqueeze(1)
    std = (consts["std_standing"][:j_dim] * standing
           + consts["std_walking"][:j_dim] * walking
           + consts["std_running"][:j_dim] * running)
    pose_err_sq = q_next * q_next
    r_pose = w_pose * torch.exp(-torch.mean(pose_err_sq / (std * std), dim=1))

    # print(r_v_xy)
    # print(r_omega_z)
    # print(r_v_z)
    # print(r_omega_xy)
    # print(r_q_tau)
    # print(r_q_ddot)
    # print(r_a_dot)
    # print(r_fa)
    # print(r_foot_c)
    # print(r_c)
    # print(r_stand_still)
    # print(r_g)
    # print("\n\n")

    return (
        r_v_xy
        + r_omega_z
        + r_v_z
        + r_omega_xy
        + r_q_tau
        + r_q_ddot
        + r_a_dot
        + r_fa
        + r_foot_c
        + r_c
        #+ r_stand_still
        + r_g
        + r_joint_pos_limits
        + r_pose
        + 0.0
    )


def main() -> None:
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    env_cfg = load_env_cfg(TASK_ID)
    env_cfg.scene.num_envs = NUM_ENVS
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=100.0)

    # Joint-order sanity check: G1 reward constants assume G1_JOINT_NAMES order.
    actual_joint_names = list(env.unwrapped.scene["robot"].joint_names)
    print("[JOINT ORDER CHECK] env:", actual_joint_names)
    print("[JOINT ORDER CHECK] expected:", G1_JOINT_NAMES)
    assert actual_joint_names == G1_JOINT_NAMES, (
        "Joint order mismatch — joint_pos_limits / pose reward tensors will be wrong"
    )

    obs = env.get_observations().to(device)
    obs_td = to_obs_tensordict(obs, device)

    obs_dim = obs_td["policy"].shape[-1]
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

    last_actions = torch.zeros(NUM_ENVS, num_actions, device=device)

    consts = setup_reward_constants(device)

    ep_return = torch.zeros(NUM_ENVS, device=device)
    ep_length = torch.zeros(NUM_ENVS, device=device, dtype=torch.long)
    finished_returns = []
    finished_lengths = []

    print(f"[INFO] task={TASK_ID} device={device} num_envs={NUM_ENVS} obs_dim={obs_dim} actions={num_actions}")
    for it in range(MAX_ITERATIONS):
        with torch.inference_mode():
            for _ in range(NUM_STEPS):


                cmd_vels = cmd_vels_from_obs(obs_td)
               
                actions = algo.act(obs_td)
                
                next_obs, rewards, dones, extras = env.step(actions.to(env.device))
                tau = env.unwrapped.scene["robot"].data.actuator_force
                ft_sensor = foot_air_time(env.unwrapped, "feet_ground_contact")
                fc_sensor = foot_contact(env.unwrapped, "feet_ground_contact")

                check_nan(next_obs, rewards, dones)


                next_obs = next_obs.to(device)
                dones = dones.to(device)
                next_obs_td = to_obs_tensordict(next_obs, device)

    
                rewards = custom_reward(
                    cmd_vels,
                    obs_td,
                    next_obs_td,
                    last_actions,
                    actions,
                    tau,
                    ft_sensor.to(device),
                    fc_sensor.to(device),
                    consts,
                )

                rewards = rewards.to(device)

                last_actions = actions.clone()

                ep_return += rewards
                ep_length += 1
                done_mask = dones.bool()
                if done_mask.any():
                    finished_returns.append(ep_return[done_mask].clone())
                    finished_lengths.append(ep_length[done_mask].clone())
                    ep_return[done_mask] = 0.0
                    ep_length[done_mask] = 0

                algo.process_env_step(next_obs_td, rewards, dones, extras)
                obs_td = next_obs_td

            algo.compute_returns(obs_td)

        loss_dict = algo.update()
        if finished_returns:
            all_r = torch.cat(finished_returns)
            all_l = torch.cat(finished_lengths)
            ep_stats = (
                f"ep_return mean={all_r.mean().item():.2f} "
                f"min={all_r.min().item():.2f} max={all_r.max().item():.2f} "
                f"ep_length mean={all_l.float().mean().item():.1f} count={all_r.numel()}"
            )
            finished_returns.clear()
            finished_lengths.clear()
        else:
            ep_stats = "no episodes finished this iter"
        print(f"Iter {it}: {ep_stats} | {loss_dict}\n")
        if it % SAVE_INTERVAL == 0:
            ckpt = save_model(algo, it)
            print(f"[INFO] saved checkpoint: {ckpt}")

    final_ckpt = save_model(algo, MAX_ITERATIONS - 1)
    print(f"[INFO] saved final checkpoint: {final_ckpt}")
    env.close()


if __name__ == "__main__":
    main()