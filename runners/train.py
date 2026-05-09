import json, time
from tqdm.notebook import trange, tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import SaveModel, CreateWorlModelInstance, get_model_name,\
                  count_parameters, SaveCkpt, LoadCkpt
import time
from storage.replay_buffer import ReplayBuffer
from utils import z_norm, minmax_norm_state_action_pair, denormalize, denormalize_z_norm
from runners.imagination import Imagination


STATE_ACTION_SPLIT = 96


def plot_on_the_fly_batch_x(x):
    """Plot `x` (B, T, state+action): batch 0; overlay action and joint slice at chosen timesteps on one x-axis."""
    if x.ndim != 3 or x.shape[-1] < STATE_ACTION_SPLIT + 1:
        return
    if x.shape[1] < 2:
        return
    action_t0 = x[0, 16, STATE_ACTION_SPLIT:].detach().float().cpu().numpy()
    joints_t1 = x[0, 17, 9:38].detach().float().cpu().numpy()

    na = int(action_t0.shape[0])
    nj = int(joints_t1.shape[0])

    fig, ax = plt.subplots(figsize=(10, 4))
    if na == nj:
        xc = np.arange(na)
        ax.plot(xc, action_t0, marker="o", markersize=3, linewidth=1, label="action x[0,38,96:]")
        ax.plot(xc, joints_t1, marker="o", markersize=3, linewidth=1, label="joints x[0,39,9:38]")
        ax.set_xlabel("shared component index (0 … n−1)")
    else:
        ax.plot(np.linspace(0.0, 1.0, na), action_t0, marker="o", markersize=3, linewidth=1, label="action x[0,38,96:]")
        ax.plot(np.linspace(0.0, 1.0, nj), joints_t1, marker="o", markersize=3, linewidth=1, label="joints x[0,39,9:38]")
        ax.set_xlabel("normalized position along each vector (lengths differ)")

    ax.set_ylabel("value")
    ax.set_title("batch 0: action and joint slice overlaid on one axes")
    ax.legend()
    fig.tight_layout()
    plt.show()


# only state-action pair
class Trainer(Imagination):

    def __init__(self, config: dict):

        self.consts = self.setup_reward_constants("cuda")

        self.config = config
        # self.data_loader = load_dataset(db_path=self.config["db_path"])

        self.scale = torch.tensor([self.config.get("scale")]).to(self.config.get("device"))
        self.offset = torch.tensor([self.config.get("offset")]).to(self.config.get("device"))

        self.mean_state_action = self.config.get("mean_state_action")
        self.std_state_action = self.config.get("std_state_action")
        self.state_mean = torch.tensor(self.mean_state_action[:96]).to(self.config.get("device"))
        self.state_std = torch.tensor(self.std_state_action[:96]).to(self.config.get("device"))
        self.action_mean = torch.tensor(self.mean_state_action[96:]).to(self.config.get("device"))
        self.action_std = torch.tensor(self.std_state_action[96:]).to(self.config.get("device"))

        self.jmin = torch.tensor([-2.5306999683380127, -0.5235999822616577, -2.7576000690460205, -0.08726699650287628, -0.8726699948310852, -0.26179999113082886, -2.5306999683380127, -2.967099905014038, -2.7576000690460205, -0.08726699650287628, -0.8726699948310852, -0.26179999113082886, -2.618000030517578, -0.5199999809265137, -0.5199999809265137, -3.089200019836426, -1.5881999731063843, -2.618000030517578, -1.0471999645233154, -1.9722199440002441, -1.6144299507141113, -1.6144299507141113, -3.089200019836426, -2.251499891281128, -2.618000030517578, -1.0471999645233154, -1.9722199440002441, -1.6144299507141113, -1.6144299507141113]).to(self.config.get("device"))
        self.jmax = torch.tensor([2.8798000812530518, 2.967099905014038, 2.7576000690460205, 2.8798000812530518, 0.5235999822616577, 0.26179999113082886, 2.8798000812530518, 0.5235999822616577, 2.7576000690460205, 2.8798000812530518, 0.5235999822616577, 0.26179999113082886, 2.618000030517578, 0.5199999809265137, 0.5199999809265137, 2.6703999042510986, 2.251499891281128, 2.618000030517578, 2.094399929046631, 1.9722199440002441, 1.6144299507141113, 1.6144299507141113, 2.6703999042510986, 1.5881999731063843, 2.618000030517578, 2.094399929046631, 1.9722199440002441, 1.6144299507141113, 1.6144299507141113]).to(self.config.get("device"))
        self.tau_min = torch.tensor([-25.0, -25.0, -25.0, -25.0, -25.0, -25.0, -25.0, -25.0, -25.0, -25.0, -88.0, -88.0, -88.0, -88.0, -88.0, -139.0, -139.0, -139.0, -139.0, -5.0, -5.0, -5.0, -5.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0]).to(self.config.get("device"))
        self.tau_max = torch.tensor([25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 88.0, 88.0, 88.0, 88.0, 88.0, 139.0, 139.0, 139.0, 139.0, 5.0, 5.0, 5.0, 5.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]).to(self.config.get("device"))

        self.replay_buffer = ReplayBuffer(dim=self.config.get("dim"), 
                                          buffer_size=self.config.get("buffer_size"), 
                                          device=self.config["device"]) 
        
        self._initialized_ = None
        self.imagination_ht = None
        self._last_imagination_state_hist = None
        self._last_imagination_action_hist = None
        self._last_imagination_st_next_pred = None
        self._last_imagination_at = None
        self.wm_initialized = None
        self.extras = dict()

    def get_model_params(self, model):
        num_model_params = 0
        for param in model.parameters():
            num_model_params += param.flatten().shape[0]
        
        print('Total params in the world model : ', num_model_params)

    def scale_policy_actions(self, policy_actions):
        return (policy_actions*self.scale) + self.offset

    def descale(self, scaled_actions):
        return (scaled_actions - self.offset) / self.scale

    def insert_into_replay_buffer(self, state_, torques, action, termination):
        # obs -> combine state and torques
        # scale the raw policy actions
        # normalize the state and action
        # obs = torch.cat((state_[..., :67], torques), dim=-1)
        obs = state_
        action = self.scale_policy_actions(action)
        obs_norm, action_norm = z_norm(obs, action, self.state_mean, self.state_std, self.action_mean, self.action_std)
        
        # state_action_pair = torch.concat([obs, action], dim=-1)
        # obs_norm, action_norm = minmax_norm_state_action_pair(state_action_pair, 
        #                                                       self.jmin,
        #                                                       self.jmax,
        #                                                       self.tau_min,
        #                                                       self.tau_max)
        
        # print("\n\n")
        
        self.replay_buffer.insert([obs_norm.to(self.config["device"]), 
                                   action_norm.to(self.config["device"]), 
                                   torch.unsqueeze(termination.to(self.config["device"]), dim=-1)])



    def on_the_fly_update(self, itr):
        # print("inside on the fly update")
        if self._initialized_ is None:
            # params
            self.M, self.N = self.config['world_model_training_params']['M'], self.config['world_model_training_params']['N']
            self.decay = self.config['world_model_training_params']['forecast_decay']
            self.state_dims = self.config['robot_params']['state_dims']
            self.action_dims = self.config['robot_params']['action_dims']
            self.num_mini_batches = self.config["world_model_training_params"]["num_mini_batches"]
            self.total_epochs = self.config['world_model_training_params']['epochs']
            self.total_grad_steps = self.num_mini_batches * self.total_epochs 

            if self.wm_initialized is None:
                # create model instance
                self.world_model = CreateWorlModelInstance(self.config)
                self.world_model.train()
                print('World Model instantiated\n\n\n')
                count_parameters(self.world_model)
                self.wm_initialized = True

            # model dir 
            model_type = "wm_gru"
            self.model_dir_name = get_model_name(model_type)
            print('this is the model name : ', self.model_dir_name)

            self._initialized_ = True
        

        # load checkpoints
        if self.config["use_ckpt"]:
            self.world_model = LoadCkpt(self.world_model,
                                   self.config["ckpt_name"],
                                   self.config["ckpt_dir"]) 

            
        

        # # get the batch data from the replay buffer and do world model update
        # idx = 1
        # for num_mb, batch in enumerate(self.replay_buffer.mini_batch_generator(sequence_length=M+N, 
        #                                                                        num_mini_batch=self.config['world_model_training_params']['num_mini_batches'], 
        #                                                                        mini_batch_size=self.config['world_model_training_params']['mini_batch_size'])):
        #     print("\n",type(batch))
        #     print("batch num : ", idx)
        #     print(num_mb)
        #     print(batch[0].shape)
        #     print(batch[1].shape)
        #     print(batch[2].shape)
        #     states = batch[0]
        #     terminations = batch[-1]
        #     print(states[0, :, :])
        #     print("\n")
        #     print(terminations[0, :, :])
        #     idx += 1
        #     x = torch.concat([batch[0], batch[1]], dim=-1)
        #     print(x.shape)
            
        #     print("\n")

        print("\n\n################")
        print("Training WM")
        training_loss = []
        self.grad_step = 1
        for epoch in range(1, self.total_epochs+1):
            # Iterate over batches
            epoch_loss = 0.0
            for mb_idx, batch in enumerate(self.replay_buffer.mini_batch_generator(sequence_length=self.M+self.N, 
                                                                                   num_mini_batch=self.config['world_model_training_params']['num_mini_batches'], 
                                                                                   mini_batch_size=self.config['world_model_training_params']['mini_batch_size'])):
                
                # print(batch.shape)
                print(f"\rGradient steps : {self.grad_step}/{self.total_grad_steps}", end="", flush=True)    

                # print(batch[-1][0,:])      

                x = torch.concat([batch[0], batch[1]], dim=-1)
                # plot_on_the_fly_batch_x(x)
                # print(x.shape)
                batch_loss = self.rnn_rollout_steps(x)
                
                epoch_loss += batch_loss
            
            training_loss.append(epoch_loss)
        print("\n################\n")


        # save model
        if itr % 10 == 0:
            model_name = f'wm-itr_{itr}.pth'
            SaveModel(self.world_model, model_name, self.model_dir_name)


        # save checkpoint
        if itr% self.config["ckpt_save_freq"] == 0:
            model_name = f"wm-itr-ckpt-epoch_{itr}.pth"
            SaveCkpt(self.world_model, model_name, self.model_dir_name, itr, epoch_loss)




    def rnn_rollout_steps(self, x):

        ht = torch.zeros((self.config['world_model_arch_params']['num_gru_layers'], 
                          x.shape[0], 
                          self.config['world_model_arch_params']['gru_hidden_dim'])).to(self.config['device'])
                
        seq_len = x.shape[1]
        batch_loss = 0
        alpha = 1.0

        # Iterate over RNN timestamps
        for t in range(seq_len-1):
            loss_t = 0
            if t < self.M-1:
                ht = self.world_model.forward(torch.unsqueeze(x[:, t, :], dim=1), 
                                            ht, predict=False) # torch.unsqueeze(x[:, t, :], dim=1) => (bs, s_dim+a_dim) -> (bs, 1, s_dim+a_dim)
            else:
                if t == self.M-1:
                    x_prev = torch.unsqueeze(x[:, t, :96], dim=1)
                    # std_logits, state_mean
                    st_next_pred, ht = self.world_model.forward(torch.unsqueeze(x[:, t, :], dim=1), ht, predict=True, x_prev=x_prev)
                else:
                    st_next_pred, ht = self.world_model.forward(torch.cat((st_next_pred, torch.unsqueeze(x[:, t, -self.action_dims:], dim=1)), dim=2), 
                                                                            ht, predict=True, x_prev=st_next_pred)
                target = torch.unsqueeze(x[:, t+1, :self.state_dims], dim=1)
                # loss_t = world_model.nll_loss(dist, target)
                # print(st_next_pred.shape)
                # print(target.shape)
                # print("\n")
                loss_t = self.world_model.mse_loss(st_pred=torch.squeeze(st_next_pred, dim=1),
                                                  st_true=torch.squeeze(target, dim=1))
                # loss_t = self.world_model.gnll_loss(state_mean=state_mean,
                #                                state_std=std_logits,
                #                                state_target=target)
                batch_loss += alpha * loss_t
                alpha *= self.decay

        batch_loss /= self.N

        # optimize
        self.world_model.optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
        self.world_model.optimizer.step()
        
        self.grad_step += 1

        return batch_loss
    

    def load_model(self, ckpt_name, ckpt_dir):
    
        
        self.M, self.N = self.config['world_model_training_params']['M'], self.config['world_model_training_params']['N']
        self.decay = self.config['world_model_training_params']['forecast_decay']
        self.state_dims = self.config['robot_params']['state_dims']
        self.action_dims = self.config['robot_params']['action_dims']
        self.num_mini_batches = self.config["world_model_training_params"]["num_mini_batches"]
        self.total_epochs = self.config['world_model_training_params']['epochs']
        self.total_grad_steps = self.num_mini_batches * self.total_epochs 

        # create model instance
        self.world_model = CreateWorlModelInstance(self.config)

        self.world_model = LoadCkpt(self.world_model,
                                   ckpt_name,
                                   ckpt_dir) 
        self.world_model.train()
        self.wm_initialized = True
    
    def _cache_imagination_last_outputs(self, state_hist, action_hist, st_next_pred, at):
        self._last_imagination_state_hist = state_hist
        self._last_imagination_action_hist = action_hist
        self._last_imagination_st_next_pred = st_next_pred
        self._last_imagination_at = at

    def _rollout_imagination_prefix(self, state_history, action_history, ht):
        """Run GRU burn-in + one predict step. Mutates nothing on ``self`` except via ``forward``."""
        x = torch.concat([state_history, action_history], dim=-1)
        seq_len = x.shape[1]
        if seq_len > 1:
            for t in range(seq_len):
                if t < seq_len - 1:
                    ht = self.world_model.forward(
                        torch.unsqueeze(x[:, t, :], dim=1),
                        ht,
                        predict=False,
                    )
                else:
                    x_prev = torch.unsqueeze(x[:, t, :96], dim=1)
                    st_mean_pred, ht = self.world_model.forward(
                        torch.unsqueeze(x[:, t, :], dim=1),
                        ht,
                        predict=True,
                        x_prev=x_prev,
                        sample=True,
                    )
                    last_action = torch.unsqueeze(x[:, -1, 96:], dim=1)
                    state_hist = torch.unsqueeze(state_history[:, -1, :], dim=1)
                    action_hist = torch.unsqueeze(action_history[:, -1, :], dim=1)
                    return state_hist, action_hist, st_mean_pred, last_action, ht
        else:
            x_prev = torch.unsqueeze(x[:, -1, :96], dim=1)
            st_mean_pred, ht = self.world_model.forward(
                torch.unsqueeze(x[:, -1, :], dim=1),
                ht,
                predict=True,
                x_prev=x_prev,
                sample=True,
            )
            last_action = torch.unsqueeze(x[:, -1, 96:], dim=1)
            return state_history, action_history, st_mean_pred, last_action, ht

    def dynamics_step(self, state_history, action_history, imagine=False):

        num_envs = state_history.shape[0]
        if self.imagination_ht is None or self.imagination_ht.shape[1] != num_envs:
            self.imagination_ht = torch.zeros(
                (
                    self.config["world_model_arch_params"]["num_gru_layers"],
                    num_envs,
                    self.config["world_model_arch_params"]["gru_hidden_dim"],
                )
            ).to(self.config["device"])

        state_hist, action_hist, st_mean_pred, last_action, self.imagination_ht = self._rollout_imagination_prefix(
            state_history, action_history, self.imagination_ht
        )
        self._cache_imagination_last_outputs(state_hist, action_hist, st_mean_pred, last_action)
        return state_hist, action_hist, st_mean_pred, last_action

    def reset(self, state_history, action_history, env_indices=None):

        # self.dynamics_step(state_history[:, :self.config["M"]-1, :], 
        #                    action_history[:, :self.config["M"]-1, :])
        
        if env_indices is None:
            return self.dynamics_step(state_history, action_history)

        idx = torch.as_tensor(env_indices, device=state_history.device)
        if idx.dtype == torch.bool:
            idx = torch.nonzero(idx, as_tuple=False).flatten()
        else:
            idx = idx.flatten().long()
        idx = torch.unique(idx)

        if idx.numel() == 0:
            if self._last_imagination_st_next_pred is not None:
                return (
                    self._last_imagination_state_hist.clone(),
                    self._last_imagination_action_hist.clone(),
                    self._last_imagination_st_next_pred.clone(),
                    self._last_imagination_at.clone(),
                )
            return self.dynamics_step(state_history, action_history)

        if self._last_imagination_st_next_pred is None or self.imagination_ht is None:
            return self.dynamics_step(state_history, action_history)

        layers = self.config["world_model_arch_params"]["num_gru_layers"]
        hidden = self.config["world_model_arch_params"]["gru_hidden_dim"]
        ht_sub = torch.zeros(
            (layers, idx.numel(), hidden),
            device=self.imagination_ht.device,
            dtype=self.imagination_ht.dtype,
        )

        sub_state = state_history.index_select(0, idx)
        sub_action = action_history.index_select(0, idx)
        sub_state_hist, sub_action_hist, sub_st, sub_at, ht_sub = self._rollout_imagination_prefix(
            sub_state, sub_action, ht_sub
        )

        self.imagination_ht.index_copy_(1, idx.to(self.imagination_ht.device), ht_sub)

        state_hist = self._last_imagination_state_hist.clone()
        action_hist = self._last_imagination_action_hist.clone()
        st_next_pred = self._last_imagination_st_next_pred.clone()
        at = self._last_imagination_at.clone()
        state_hist.index_copy_(0, idx, sub_state_hist)
        action_hist.index_copy_(0, idx, sub_action_hist)
        st_next_pred.index_copy_(0, idx, sub_st)
        at.index_copy_(0, idx, sub_at)
        self._cache_imagination_last_outputs(state_hist, action_hist, st_next_pred, at)
        return state_hist, action_hist, st_next_pred, at

    def process_imagined_obs(self, denormalized_obs, last_cmd_vels, denormalized_last_actions):
        imagined_obs = torch.concat([denormalized_obs[:, :67], 
                                     denormalized_last_actions, 
                                     last_cmd_vels], dim=-1)
        return imagined_obs


    def get_imagined_obs(self, state_history, action_history, last_cmd_vels):
        x = torch.concat([state_history, action_history], dim=-1)
        # seq_len = x.shape[1]
        # num_envs = x.shape[0]

        x_curr = torch.unsqueeze(x[:, -1, :96], dim=1)
        st_pred, self.imagination_ht = self.world_model.forward(torch.unsqueeze(x[:, -1, :], dim=1),
                                                                self.imagination_ht,
                                                                predict=True,
                                                                x_prev=x_curr,
                                                                sample=True)
        
        # # denormalize predicted state and last-step normalized joint targets
        last_action = torch.unsqueeze(x[:, -1, 96:], dim=1)


        return st_pred, last_action
    

    def preprocess_obs(self, st_next_pred, last_action, last_cmd_vels):
        # denormalized_obs, last_actions_raw = denormalize(
        #                                                 st_next_pred,
        #                                                 self.jmin,
        #                                                 self.jmax,
        #                                                 self.tau_min,
        #                                                 self.tau_max,
        #                                                 last_action=last_action,
        #                                             )
        # print(st_next_pred.shape)
        
        denormalized_obs, last_actions_raw = denormalize_z_norm(st_next_pred,
                                                                last_action,
                                                                self.state_mean,
                                                                self.state_std,
                                                                self.action_mean,
                                                                self.action_std)
        # print(denormalized_obs.shape)
        # descale the actions to match policy dist
        last_actions_raw = self.descale(last_actions_raw)

        # print(last_action.shape)

        imagined_obs = self.process_imagined_obs(torch.squeeze(st_next_pred, dim=1),
                                                 last_cmd_vels,
                                                 torch.squeeze(last_action, dim=1))
        
        return imagined_obs, last_actions_raw
        

    def imagination_step(self, st_prev, imagined_action_t, at_prev, last_cmd_vel,
                                ep_len, max_ep_len, prev_state_hist):
        
        # scale the policy output actions
        scaled_imagined_actions_t = self.scale_policy_actions(imagined_action_t)

        scaled_imagined_actions_prev = self.scale_policy_actions(at_prev)

        # normalize states and actions
        # state_action_pair = torch.concat([torch.squeeze(st_prev, dim=1), scaled_imagined_actions_t], dim=-1)
        # obs_norm, action_norm = minmax_norm_state_action_pair(state_action_pair, 
        #                                                       self.jmin,
        #                                                       self.jmax,
        #                                                       self.tau_min,
        #                                                       self.tau_max)
        
        obs_norm, action_norm = z_norm(st_prev, scaled_imagined_actions_t, self.state_mean, self.state_std, self.action_mean, self.action_std)

        state_hist = st_prev
        action_hist = torch.unsqueeze(action_norm, dim=1)

        # do dynamics step
        state_hist, action_hist, st_next_pred, at = self.dynamics_step(state_hist, action_hist)
        
        # # denormalize your st_next_pred 
        # imagined_obs_t_next = denormalize(st_next_pred,
        #                                   self.jmin,
        #                                   self.jmax,
        #                                   self.tau_min,
        #                                   self.tau_max,)
        
        # # denormalize prev state 
        # denormalized_prev_state = denormalize(torch.squeeze(prev_state_hist, dim=1),
        #                                     self.jmin,
        #                                     self.jmax,
        #                                     self.tau_min,
        #                                     self.tau_max,)
        
    

        # compute rewards
        # rewards = self.custom_rewards(last_cmd_vel, 
        #                               torch.squeeze(prev_state_hist, dim=1), 
        #                               torch.squeeze(st_next_pred, dim=1), 
        #                               at_prev, 
        #                               imagined_action_t,
        #                               self.consts)

        denormalized_obs, _ = denormalize_z_norm(st_next_pred,
                                                at,
                                                self.state_mean,
                                                self.state_std,
                                                self.action_mean,
                                                self.action_std)

        rewards = self.custom_rewards(last_cmd_vel, 
                                      torch.squeeze(st_prev, dim=1), 
                                      torch.squeeze(st_next_pred, dim=1), 
                                      scaled_imagined_actions_prev, 
                                      scaled_imagined_actions_t,
                                      self.consts)

        # compute dones
        dones, timeouts = self.custom_dones(next_obs_td=torch.squeeze(st_next_pred, dim=1),
                                            episode_lengths=ep_len,
                                            max_episode_length=max_ep_len)
        


        imagined_obs_next, _ = self.preprocess_obs(st_next_pred,
                                                  at,
                                                  last_cmd_vel)
        

        
        self.extras["time_outs"] = timeouts

        return state_hist, action_hist, st_next_pred, at, imagined_obs_next, rewards, dones, self.extras


    def get_warm_start_buffer(self,):
        mini_batch_itr = self.replay_buffer.mini_batch_generator(32,
                                                             1,
                                                             self.config.get("NUM_ENVS"))
        state_hist_batch, action_hist_batch = next(mini_batch_itr)[:2]
        
        return state_hist_batch, action_hist_batch
    

    def custom_step(self,  cmd_vels, obs_td, next_obs_td, last_actions, actions, episode_lengths, max_episode_length):

        rewards = self.custom_rewards(cmd_vels, obs_td, next_obs_td, last_actions, actions, self.consts)
        dones, timeouts = self.custom_dones(next_obs_td, episode_lengths, max_episode_length)
        self.extras["time_outs"] = timeouts
        return rewards, dones, self.extras



    