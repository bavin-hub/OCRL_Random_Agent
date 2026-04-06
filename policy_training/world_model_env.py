import torch
import math
from mjlab.managers.scene_entity_config import SceneEntityCfg
from data.preprocessor import load_dataset
# We rely on some math from mdp if we were parsing it exactly, but we can re-implement
# the tracking errors in native PyTorch directly here.

class WMSlice:
    # Based on base.py: state_vec = [base_lin (3), base_ang (3), grav_proj (3), jpos (29), jvel (29), jtorque (29)]
    LIN_VEL = slice(0, 3)
    ANG_VEL = slice(3, 6)
    PROJ_GRAV = slice(6, 9)
    JOINT_POS = slice(9, 38)
    JOINT_VEL = slice(38, 67)
    JOINT_TORQUE = slice(67, 96)

class WorldModelEnv:
    def __init__(self, cfg, world_model, db_dir_name, device="cuda"):
        self.cfg = cfg
        self.world_model = world_model
        world_model.eval()
        self.device = device
        
        self.num_envs = cfg.scene.num_envs
        self.max_episode_length = int(cfg.episode_length_s / cfg.sim.mujoco.timestep / cfg.decimation) # typically 1000
        self.step_dt = cfg.sim.mujoco.timestep * cfg.decimation

        self.num_actions = 29
        self.num_obs = 3 + 3 + 3 + 29 + 29 + 29 # 96
        
        self.step_counts = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Load dataset for resets
        db_path = f"data/{db_dir_name}" # You might want to grab this cleanly from args
        print(f"[WorldModelEnv] Loading pretraining dataset from {db_path} for resets...")
        
        # M=32, N=8 matches config, we just need M steps to warm up the GRU
        self.M = 32
        import configparser
        with open("config.json", "r") as f:
            import json
            app_cfg = json.load(f)
            self.M = app_cfg['world_model_training_params']['M']
            self.base_cfg = app_cfg

        # Grab dataset chunk provider
        self.dataset = load_dataset(db_paths=[db_path + "/combined_transitions.db"], 
                                    batch_size=self.num_envs, 
                                    M=self.M, N=0,
                                    run_mode="train")
        # We will manually sample chunks from the `all_windows` of the dataset.
        
        # Current state trackers
        self.obs_norm = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.ht = None
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.commands = torch.zeros((self.num_envs, 3), device=self.device) # linear x, y, angular z

        # Get bounds config for un-normalization (you will need these accurate bounds)
        self.lin_vel_limit = 6.0
        self.ang_vel_limit = 12.0
        self.grav_limit = 9.81
        self.joint_vel_limit = 20.0
        
        # For simplicity, we assume tracking is 0 momentarily without full command manager
        # But command manager resampling can be added here easily.

    @property
    def unwrapped(self):
        return self

    def _sample_commands(self, num_envs=None):
        n = self.num_envs if num_envs is None else num_envs
        # Mimic commands generation (UniformVelocityCommandCfg in cfg): x: [-1, 2], y: [-1,1], yaw: [-1,1]
        cmds = torch.zeros((n, 3), device=self.device)
        cmds[:, 0].uniform_(-1.0, 2.0)
        cmds[:, 1].uniform_(-1.0, 1.0)
        cmds[:, 2].uniform_(-1.0, 1.0)
        return cmds

    def _unnormalize(self, x_norm):
        """Un-normalizes the world model state back to physical units."""
        # Using symmetric bounds matching base.py
        phys_lin_vel = x_norm[:, WMSlice.LIN_VEL] * self.lin_vel_limit
        phys_ang_vel = x_norm[:, WMSlice.ANG_VEL] * self.ang_vel_limit
        phys_grav = x_norm[:, WMSlice.PROJ_GRAV] * self.grav_limit
        
        # For joint limits, assuming default G1 ranges or approximate +/- PI for simplicity
        # Ideally, pull exact jmin/jmax from MJCF
        phys_jpos = x_norm[:, WMSlice.JOINT_POS] * 3.14 
        phys_jvel = x_norm[:, WMSlice.JOINT_VEL] * self.joint_vel_limit
        
        return phys_lin_vel, phys_ang_vel, phys_grav, phys_jpos, phys_jvel
        
    def get_observations(self):
        phys_lin_vel, phys_ang_vel, phys_grav, phys_jpos, phys_jvel = self._unnormalize(self.obs_norm)
        
        # Reconstruct actor_terms matching velocity_env_cfg
        # "base_ang_vel", "projected_gravity", "command", "joint_pos", "joint_vel", "actions"
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
        if num_resets == 0: return self.get_observations()
        
        import random
        # Sample random chunks from the DB for burn-in
        indices = random.sample(range(len(self.dataset.all_windows)), num_resets)
        chunks = [self.dataset.all_windows[i] for i in indices]
        
        # Each chunk is [M+N, 96 + action_dims]
        # We need the first M steps for burn-in
        chunk_tensors = torch.tensor(chunks, dtype=torch.float32, device=self.device) # [num_resets, M, 96+A]
        
        # Burn-in pass through GRU
        ht = torch.zeros((self.base_cfg['world_model_arch_params']['num_gru_layers'], 
                          num_resets, 
                          self.base_cfg['world_model_arch_params']['gru_hidden_dim']), device=self.device)
        
        with torch.no_grad():
            for t in range(self.M - 1):
                x = chunk_tensors[:, t, :] # [num_resets, 96+29]
                ht = self.world_model.forward(x.unsqueeze(1), ht, predict=False)
            
            # The last step prediction
            x_prev = chunk_tensors[:, self.M-1, :96].unsqueeze(1)
            x_step = chunk_tensors[:, self.M-1, :].unsqueeze(1)
            st_next_pred, ht, _, _ = self.world_model.forward(x_step, ht, predict=True, x_prev=x_prev)
        
        # Inject the warmed-up state back into the master buffers
        self.obs_norm[env_ids] = st_next_pred.squeeze(1).float()
        
        if self.ht is None:
            self.ht = torch.zeros((self.base_cfg['world_model_arch_params']['num_gru_layers'], 
                                   self.num_envs, 
                                   self.base_cfg['world_model_arch_params']['gru_hidden_dim']), device=self.device)
        self.ht[:, env_ids, :] = ht.float()
        
        self.commands[env_ids] = self._sample_commands(num_resets)
        self.step_counts[env_ids] = 0
        self.actions[env_ids] = 0.0
        
        return self.get_observations()

    def _compute_rewards(self):
        phys_lin_vel, phys_ang_vel, phys_grav, phys_jpos, phys_jvel = self._unnormalize(self.obs_norm)
        
        # 1. track_linear_velocity
        xy_err = torch.sum(torch.square(self.commands[:, :2] - phys_lin_vel[:, :2]), dim=1)
        z_err = torch.square(phys_lin_vel[:, 2])
        lin_vel_error = xy_err + (2 * z_err)
        rew_lin_vel = torch.exp(-lin_vel_error / 0.25) * 1.0 
        
        # 2. track_angular_velocity
        z_err_ang = torch.square(self.commands[:, 2] - phys_ang_vel[:, 2])
        xy_err_ang = torch.sum(torch.square(phys_ang_vel[:, :2]), dim=1)
        ang_vel_error = z_err_ang + (0.05 * xy_err_ang)
        rew_ang_vel = torch.exp(-ang_vel_error / 0.5) * 1.0 
        
        # 3. body_orientation_l2
        xy_squared = torch.sum(torch.square(phys_grav[:, :2]), dim=1)
        rew_body_orient = xy_squared * -1.0
        
        # 4. body_ang_vel penalty
        ang_vel_xy = phys_ang_vel[:, :2]
        rew_body_ang_vel = torch.sum(torch.square(ang_vel_xy), dim=1) * -0.05
        
        total_reward = rew_lin_vel + rew_ang_vel + rew_body_orient + rew_body_ang_vel
        return total_reward

    def _compute_dones(self):
        # Timeouts
        timeouts = self.step_counts >= self.max_episode_length
        
        # Orientation fell_over (e.g. angle > 70 deg)
        phys_lin_vel, phys_ang_vel, phys_grav, phys_jpos, phys_jvel = self._unnormalize(self.obs_norm)
        # If gravity Z is highly negative (robot upside down) or xy projection too high:
        fell_over = phys_grav[:, 2] > -math.cos(math.radians(70.0))
        
        return timeouts | fell_over, timeouts

    def step(self, actions_tensor):
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
