import torch
import math

class Imagination:

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
        


    
    def _resolve_std(self, joint_names, std_map):
        out = []
        for name in joint_names:
            key = name.replace("left_", "").replace("right_", "").replace("_joint", "")
            out.append(std_map.get(key, 0.1))
        return out


    def setup_reward_constants(self, device):
        default_jp = torch.tensor(self.G1_DEFAULT_JOINT_POS, device=device, dtype=torch.float32)
        lo_abs = torch.tensor(self.G1_SOFT_JOINT_LIMITS_LO, device=device, dtype=torch.float32)
        hi_abs = torch.tensor(self.G1_SOFT_JOINT_LIMITS_HI, device=device, dtype=torch.float32)
        # Pre-shift limits into the relative joint-pos frame used by the actor obs
        # (obs gives q - q_default, so compare against limits in the same frame).

        G1_STD_WALKING = self._resolve_std(self.G1_JOINT_NAMES, self._WALKING_STD_MAP)
        G1_STD_RUNNING = self._resolve_std(self.G1_JOINT_NAMES, self._RUNNING_STD_MAP)

        return {
            "soft_limits_lo_rel": lo_abs - default_jp,
            "soft_limits_hi_rel": hi_abs - default_jp,
            "std_standing": torch.tensor(self.G1_STD_STANDING, device=device, dtype=torch.float32),
            "std_walking": torch.tensor(G1_STD_WALKING, device=device, dtype=torch.float32),
            "std_running": torch.tensor(G1_STD_RUNNING, device=device, dtype=torch.float32),
        }


    def custom_rewards(self, cmd_vels, obs_td, next_obs_td, last_actions, actions, consts):

        # print(cmd_vels.shape)
        # print(obs_td.shape)
        # print(next_obs_td.shape)
        # print(last_actions.shape)
        # print(actions.shape)
        # print("\n\n\n")
        
        w_v_xy = 2.0
        w_omega_z = 1.0
        w_omega_xy = -0.05
        w_q_ddot = -2.5e-7
        w_a_dot = -0.05
        w_g = -1.0
        w_joint_pos_limits = -10.0
        w_pose = 1.0

        # w_v_xy = 1.0
        # w_omega_z = 1.0
        # w_omega_xy = -0.02
        # w_q_ddot = -2.5e-7
        # w_a_dot = -0.01
        # w_g = -1.0
        # w_joint_pos_limits = -0.1
        # w_pose = 1.0

        step_dt = 0.02

        policy = obs_td
        next_policy = next_obs_td
        n = policy.shape[0]
        dtype = policy.dtype

        a_dim = actions.shape[-1]
        tail = policy.shape[1] - 12 - a_dim - 3
        j_dim = tail // 2

        v_xy = next_policy[:, 0:2]
        vz = next_policy[:, 2:3]
        omega_xy = next_policy[:, 3:5]
        omega_z = next_policy[:, 5:6]
        g_xy = next_policy[:, 6:8]
        q_next = next_policy[:, 9 : 9 + j_dim]
        q_vel = policy[:, 9 + j_dim : 9 + 2 * j_dim]
        q_vel_next = next_policy[:, 9 + j_dim : 9 + 2 * j_dim]
        q_ddot = (q_vel_next - q_vel) / step_dt

        c_xy = cmd_vels[:, :2]
        cz_f = cmd_vels[:, 2:3].reshape(n, -1).squeeze(-1)
        oz_f = omega_z.reshape(n, -1).squeeze(-1)
        vz_f = vz.reshape(n, -1).squeeze(-1)

        # Track lin vel (xy + z folded into the same exp; world_model_env style).
        d_xy = c_xy - v_xy
        xy_err_sum = (d_xy * d_xy).sum(dim=-1)
        z_err_sum = vz_f * vz_f
        r_v_xy = w_v_xy * torch.exp(-(xy_err_sum + z_err_sum) / 0.25)

        # Track ang vel z.
        d_z = cz_f - oz_f
        r_omega_z = w_omega_z * torch.exp(-(d_z * d_z) / 0.5)

        # Penalties: roll/pitch rates, joint accel, action rate, flat orientation.
        r_omega_xy = w_omega_xy * (omega_xy * omega_xy).sum(dim=-1)
        r_q_ddot = w_q_ddot * (q_ddot * q_ddot).sum(dim=-1)
        r_a_dot = w_a_dot * ((actions - last_actions) ** 2).sum(dim=-1)
        r_g = w_g * (g_xy * g_xy).sum(dim=-1)

        # Joint soft-limit penalty. q_next is in (q_abs - q_default) frame; limits
        # are pre-shifted to match.
        lo_rel = consts["soft_limits_lo_rel"][:j_dim]
        hi_rel = consts["soft_limits_hi_rel"][:j_dim]
        below = -(q_next - lo_rel).clamp(max=0.0)
        above = (q_next - hi_rel).clamp(min=0.0)
        r_joint_pos_limits = w_joint_pos_limits * torch.sum(below + above, dim=1)

        # Speed-dependent posture tracking. q_next is already (q_abs - q_default),
        # so the per-joint squared error is just q_next**2.
        total_speed = torch.norm(cmd_vels[:, :2], dim=1) + torch.abs(cmd_vels[:, 2])
        standing = (total_speed < 0.5).to(dtype=dtype).unsqueeze(1)
        walking = ((total_speed >= 0.5) & (total_speed < 1.5)).to(dtype=dtype).unsqueeze(1)
        running = (total_speed >= 1.5).to(dtype=dtype).unsqueeze(1)
        std = (consts["std_standing"][:j_dim] * standing
            + consts["std_walking"][:j_dim] * walking
            + consts["std_running"][:j_dim] * running)
        r_pose = w_pose * torch.exp(-torch.mean((q_next * q_next) / (std * std), dim=1))

        total_reward = (
            r_v_xy
            + r_omega_z
            + r_omega_xy
            + r_q_ddot
            + r_a_dot
            + r_g
            + r_joint_pos_limits
            + r_pose
        )

        # print("r_v_xy : ", r_v_xy)
        # print("r_omega_z : ", r_omega_z)
        # print("r_omega_xy : ", r_omega_xy)
        # print("r_q_ddot : ", r_q_ddot)
        # print("r_a_dot : ", r_a_dot)
        # print("r_g : ", r_g)
        # print("r_joint_pos_limits : ", r_joint_pos_limits)
        # print("r_pose : ", r_pose)

        # print("rewards : ", total_reward)
        # print("\n\n")

        return total_reward * 0.5
        
        

    def custom_dones(self, next_obs_td, episode_lengths, max_episode_length, fell_over_limit_rad=math.radians(70.0)):
        g_xy = next_obs_td[:, 6:8]
        fell_over = (g_xy * g_xy).sum(dim=-1) > math.sin(fell_over_limit_rad) ** 2
        time_outs = episode_lengths >= max_episode_length
        dones = (time_outs | fell_over).to(dtype=torch.bool)
        # print("dones : ", dones.shape)
        return dones, time_outs.to(dtype=torch.bool)

    # def _imagination_step(self, cmd_vels, obs_td, next_obs_td, last_actions, 
    #                             actions, consts, ep_len, max_ep_len):
    #     return self.custom_rewards(cmd_vels, obs_td, next_obs_td, last_actions, actions, consts), self.custom_dones(next_obs_td, ep_len, max_ep_len)
    
    # def reset(self, dones):
    #     pass