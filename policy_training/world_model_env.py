import torch
import math
from mjlab.managers.scene_entity_config import SceneEntityCfg
from data.preprocessor import load_dataset


class WMSlice:
    # Based on base.py: state_vec = [base_lin (3), base_ang (3), grav_proj (3), jpos (29), jvel (29), jtorque (29)]
    LIN_VEL = slice(0, 3)
    ANG_VEL = slice(3, 6)
    PROJ_GRAV = slice(6, 9)
    JOINT_POS = slice(9, 38)
    JOINT_VEL = slice(38, 67)
    JOINT_TORQUE = slice(67, 96)


# G1 joint ordering (from MJCF, verified via extract_g1_joint_info.py).
G1_JOINT_NAMES = [
    "left_hip_pitch_joint",     # 0
    "left_hip_roll_joint",      # 1
    "left_hip_yaw_joint",       # 2
    "left_knee_joint",          # 3
    "left_ankle_pitch_joint",   # 4
    "left_ankle_roll_joint",    # 5
    "right_hip_pitch_joint",    # 6
    "right_hip_roll_joint",     # 7
    "right_hip_yaw_joint",      # 8
    "right_knee_joint",         # 9
    "right_ankle_pitch_joint",  # 10
    "right_ankle_roll_joint",   # 11
    "waist_yaw_joint",          # 12
    "waist_roll_joint",         # 13
    "waist_pitch_joint",        # 14
    "left_shoulder_pitch_joint",  # 15
    "left_shoulder_roll_joint",   # 16
    "left_shoulder_yaw_joint",    # 17
    "left_elbow_joint",           # 18
    "left_wrist_roll_joint",      # 19
    "left_wrist_pitch_joint",     # 20
    "left_wrist_yaw_joint",       # 21
    "right_shoulder_pitch_joint", # 22
    "right_shoulder_roll_joint",  # 23
    "right_shoulder_yaw_joint",   # 24
    "right_elbow_joint",          # 25
    "right_wrist_roll_joint",     # 26
    "right_wrist_pitch_joint",    # 27
    "right_wrist_yaw_joint",      # 28
]

# From HOME_KEYFRAME in g1_constants.py, mapped to joint ordering above.
G1_DEFAULT_JOINT_POS = [
    -0.1,   # left_hip_pitch
     0.0,   # left_hip_roll
     0.0,   # left_hip_yaw
     0.3,   # left_knee
    -0.2,   # left_ankle_pitch
     0.0,   # left_ankle_roll
    -0.1,   # right_hip_pitch
     0.0,   # right_hip_roll
     0.0,   # right_hip_yaw
     0.3,   # right_knee
    -0.2,   # right_ankle_pitch
     0.0,   # right_ankle_roll
     0.0,   # waist_yaw
     0.0,   # waist_roll
     0.0,   # waist_pitch
     0.35,  # left_shoulder_pitch
     0.18,  # left_shoulder_roll
     0.0,   # left_shoulder_yaw
     0.87,  # left_elbow
     0.0,   # left_wrist_roll
     0.0,   # left_wrist_pitch
     0.0,   # left_wrist_yaw
     0.35,  # right_shoulder_pitch
    -0.18,  # right_shoulder_roll
     0.0,   # right_shoulder_yaw
     0.87,  # right_elbow
     0.0,   # right_wrist_roll
     0.0,   # right_wrist_pitch
     0.0,   # right_wrist_yaw
]

# Soft joint limits (90% of range, matching soft_joint_pos_limit_factor=0.9).
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

# Per-joint std for variable_posture reward (from G1 env config).
# Standing: tight tolerance.
G1_STD_STANDING = [0.05] * 29

# Walking: per-joint group tolerances.
_WALKING_STD_MAP = {
    "hip_pitch": 0.5, "hip_roll": 0.15, "hip_yaw": 0.15,
    "knee": 0.5, "ankle_pitch": 0.15, "ankle_roll": 0.1,
    "waist_yaw": 0.15, "waist_roll": 0.1, "waist_pitch": 0.1,
    "shoulder_pitch": 0.15, "shoulder_roll": 0.1, "shoulder_yaw": 0.1,
    "elbow": 0.1, "wrist_roll": 0.1, "wrist_pitch": 0.1, "wrist_yaw": 0.1,
}

# Running: per-joint group tolerances.
_RUNNING_STD_MAP = {
    "hip_pitch": 0.5, "hip_roll": 0.25, "hip_yaw": 0.25,
    "knee": 0.5, "ankle_pitch": 0.25, "ankle_roll": 0.1,
    "waist_yaw": 0.25, "waist_roll": 0.1, "waist_pitch": 0.1,
    "shoulder_pitch": 0.25, "shoulder_roll": 0.1, "shoulder_yaw": 0.1,
    "elbow": 0.1, "wrist_roll": 0.1, "wrist_pitch": 0.1, "wrist_yaw": 0.1,
}


def _resolve_std(joint_names, std_map):
    """Map joint names to std values using substring matching."""
    result = []
    for name in joint_names:
        # Strip left_/right_ prefix and _joint suffix for matching.
        key = name.replace("left_", "").replace("right_", "").replace("_joint", "")
        result.append(std_map.get(key, 0.1))
    return result


G1_STD_WALKING = _resolve_std(G1_JOINT_NAMES, _WALKING_STD_MAP)
G1_STD_RUNNING = _resolve_std(G1_JOINT_NAMES, _RUNNING_STD_MAP)


class WorldModelEnv:
    def __init__(self, cfg, world_model, db_dir_name, device="cuda"):
        self.cfg = cfg
        self.world_model = world_model
        world_model.eval()
        self.device = device

        self.num_envs = cfg.scene.num_envs
        self.max_episode_length = int(cfg.episode_length_s / cfg.sim.mujoco.timestep / cfg.decimation)
        self.step_dt = cfg.sim.mujoco.timestep * cfg.decimation

        self.num_actions = 29
        self.num_obs = 3 + 3 + 3 + 29 + 29 + 29  # 96

        self.step_counts = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Load dataset for resets
        db_path = f"data/{db_dir_name}"
        print(f"[WorldModelEnv] Loading pretraining dataset from {db_path} for resets...")

        self.M = 32
        import configparser
        with open("config.json", "r") as f:
            import json
            app_cfg = json.load(f)
            self.M = app_cfg['world_model_training_params']['M']
            self.base_cfg = app_cfg

        self.dataset = load_dataset(db_paths=[db_path + "/combined_transitions.db"],
                                    batch_size=self.num_envs,
                                    M=self.M, N=0,
                                    run_mode="train")

        # Current state trackers
        self.obs_norm = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.ht = None
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.prev_joint_vel = torch.zeros((self.num_envs, 29), device=self.device)
        self.commands = torch.zeros((self.num_envs, 3), device=self.device)

        # Un-normalization bounds
        self.lin_vel_limit = 6.0
        self.ang_vel_limit = 12.0
        self.grav_limit = 9.81
        self.joint_vel_limit = 20.0

        # G1 constants as tensors
        self.default_joint_pos = torch.tensor(G1_DEFAULT_JOINT_POS, device=device, dtype=torch.float32)
        self.soft_limits_lo = torch.tensor(G1_SOFT_JOINT_LIMITS_LO, device=device, dtype=torch.float32)
        self.soft_limits_hi = torch.tensor(G1_SOFT_JOINT_LIMITS_HI, device=device, dtype=torch.float32)
        self.std_standing = torch.tensor(G1_STD_STANDING, device=device, dtype=torch.float32)
        self.std_walking = torch.tensor(G1_STD_WALKING, device=device, dtype=torch.float32)
        self.std_running = torch.tensor(G1_STD_RUNNING, device=device, dtype=torch.float32)

    @property
    def unwrapped(self):
        return self

    def _sample_commands(self, num_envs=None):
        n = self.num_envs if num_envs is None else num_envs
        cmds = torch.zeros((n, 3), device=self.device)
        cmds[:, 0].uniform_(-1.0, 2.0)
        cmds[:, 1].uniform_(-1.0, 1.0)
        cmds[:, 2].uniform_(-1.0, 1.0)
        return cmds

    def _unnormalize(self, x_norm):
        """Un-normalizes the world model state back to physical units."""
        phys_lin_vel = x_norm[:, WMSlice.LIN_VEL] * self.lin_vel_limit
        phys_ang_vel = x_norm[:, WMSlice.ANG_VEL] * self.ang_vel_limit
        phys_grav = x_norm[:, WMSlice.PROJ_GRAV] * self.grav_limit
        phys_jpos = x_norm[:, WMSlice.JOINT_POS] * 3.14
        phys_jvel = x_norm[:, WMSlice.JOINT_VEL] * self.joint_vel_limit
        return phys_lin_vel, phys_ang_vel, phys_grav, phys_jpos, phys_jvel

    def get_observations(self):
        phys_lin_vel, phys_ang_vel, phys_grav, phys_jpos, phys_jvel = self._unnormalize(self.obs_norm)

        actor_obs = torch.cat([
            phys_ang_vel,
            phys_grav,
            self.commands,
            phys_jpos,
            phys_jvel,
            self.actions
        ], dim=-1)

        critic_obs = torch.cat([
            actor_obs,
            phys_lin_vel
        ], dim=-1)

        return {"actor": actor_obs, "critic": critic_obs}

    def reset(self):
        return self._reset_envs(torch.arange(self.num_envs, device=self.device))

    def _reset_envs(self, env_ids):
        num_resets = len(env_ids)
        if num_resets == 0:
            return self.get_observations()

        import random
        indices = random.sample(range(len(self.dataset.all_windows)), num_resets)
        chunks = [self.dataset.all_windows[i] for i in indices]
        chunk_tensors = torch.tensor(chunks, dtype=torch.float32, device=self.device)

        ht = torch.zeros((self.base_cfg['world_model_arch_params']['num_gru_layers'],
                          num_resets,
                          self.base_cfg['world_model_arch_params']['gru_hidden_dim']), device=self.device)

        with torch.no_grad():
            for t in range(self.M - 1):
                x = chunk_tensors[:, t, :]
                ht = self.world_model.forward(x.unsqueeze(1), ht, predict=False)

            x_prev = chunk_tensors[:, self.M-1, :96].unsqueeze(1)
            x_step = chunk_tensors[:, self.M-1, :].unsqueeze(1)
            st_next_pred, ht, _, _ = self.world_model.forward(x_step, ht, predict=True, x_prev=x_prev)

        self.obs_norm[env_ids] = st_next_pred.squeeze(1).float()

        if self.ht is None:
            self.ht = torch.zeros((self.base_cfg['world_model_arch_params']['num_gru_layers'],
                                   self.num_envs,
                                   self.base_cfg['world_model_arch_params']['gru_hidden_dim']), device=self.device)
        self.ht[:, env_ids, :] = ht.float()

        self.commands[env_ids] = self._sample_commands(num_resets)
        self.step_counts[env_ids] = 0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.prev_joint_vel[env_ids] = 0.0

        return self.get_observations()

    def _compute_rewards(self):
        phys_lin_vel, phys_ang_vel, phys_grav, phys_jpos, phys_jvel = self._unnormalize(self.obs_norm)

        # 1. track_linear_velocity (w=1.0, std=sqrt(0.25))
        xy_err = torch.sum(torch.square(self.commands[:, :2] - phys_lin_vel[:, :2]), dim=1)
        z_err = torch.square(phys_lin_vel[:, 2])
        lin_vel_error = xy_err + z_err
        rew_lin_vel = torch.exp(-lin_vel_error / 0.25) * 1.0

        # 2. track_angular_velocity (w=1.0, std=sqrt(0.5))
        z_err_ang = torch.square(self.commands[:, 2] - phys_ang_vel[:, 2])
        rew_ang_vel = torch.exp(-z_err_ang / 0.5) * 1.0

        # 3. body_orientation_l2 (w=-1.0)
        rew_body_orient = torch.sum(torch.square(phys_grav[:, :2]), dim=1) * -1.0

        # 4. body_ang_vel penalty (w=-0.05)
        rew_body_ang_vel = torch.sum(torch.square(phys_ang_vel[:, :2]), dim=1) * -0.05

        # 5. action_rate_l2 (w=-0.05)
        rew_action_rate = torch.sum(torch.square(self.actions - self.prev_actions), dim=1) * -0.05

        # 6. joint_acc_l2 (w=-2.5e-7)
        joint_acc = (phys_jvel - self.prev_joint_vel) / self.step_dt
        rew_joint_acc = torch.sum(torch.square(joint_acc), dim=1) * -2.5e-7

        # 7. joint_pos_limits (w=-10.0)
        below = -(phys_jpos - self.soft_limits_lo).clamp(max=0.0)
        above = (phys_jpos - self.soft_limits_hi).clamp(min=0.0)
        out_of_limits = below + above
        rew_joint_pos_limits = torch.sum(out_of_limits, dim=1) * -10.0

        # 8. pose / variable_posture (w=1.0)
        total_speed = torch.norm(self.commands[:, :2], dim=1) + torch.abs(self.commands[:, 2])
        standing_mask = (total_speed < 0.1).float()
        walking_mask = ((total_speed >= 0.1) & (total_speed < 1.5)).float()
        running_mask = (total_speed >= 1.5).float()
        std = (self.std_standing * standing_mask.unsqueeze(1)
               + self.std_walking * walking_mask.unsqueeze(1)
               + self.std_running * running_mask.unsqueeze(1))
        pose_err_sq = torch.square(phys_jpos - self.default_joint_pos)
        rew_pose = torch.exp(-torch.mean(pose_err_sq / (std ** 2), dim=1)) * 1.0

        # 9. stand_still (w=-1.0)
        diff_angle_sq = torch.sum(torch.square(phys_jpos - self.default_joint_pos), dim=1)
        total_cmd = torch.norm(self.commands[:, :2], dim=1) + torch.abs(self.commands[:, 2])
        stand_scale = (total_cmd <= 0.1).float()
        rew_stand_still = diff_angle_sq * stand_scale * -1.0

        total_reward = (rew_lin_vel + rew_ang_vel + rew_body_orient + rew_body_ang_vel
                        + rew_action_rate + rew_joint_acc + rew_joint_pos_limits
                        + rew_pose + rew_stand_still)
        return total_reward

    def _compute_dones(self):
        timeouts = self.step_counts >= self.max_episode_length
        phys_lin_vel, phys_ang_vel, phys_grav, phys_jpos, phys_jvel = self._unnormalize(self.obs_norm)
        fell_over = phys_grav[:, 2] > -math.cos(math.radians(70.0))
        return timeouts | fell_over, timeouts

    def step(self, actions_tensor):
        self.prev_actions = self.actions.clone()
        _, _, _, _, phys_jvel_before = self._unnormalize(self.obs_norm)
        self.prev_joint_vel = phys_jvel_before.clone()

        self.actions = actions_tensor
        x = torch.cat([self.obs_norm, self.actions], dim=-1).unsqueeze(1)

        with torch.no_grad():
            st_next_pred, self.ht, _, _ = self.world_model.forward(x, self.ht, predict=True, x_prev=self.obs_norm.unsqueeze(1))

        self.obs_norm = st_next_pred.squeeze(1).float()
        self.step_counts += 1

        rewards = self._compute_rewards()
        dones, timeouts = self._compute_dones()

        if dones.any():
            env_ids = dones.nonzero(as_tuple=False).flatten()
            self._reset_envs(env_ids)

        extras = {"time_outs": timeouts}
        return self.get_observations(), rewards, dones, extras

    def close(self):
        pass
