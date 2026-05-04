import json, time
from tqdm.notebook import trange, tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import SaveModel, CreateWorlModelInstance, get_model_name,\
                  count_parameters, SaveCkpt, LoadCkpt
import time
from storage.replay_buffer import ReplayBuffer
from utils import z_norm, minmax_norm_state_action_pair

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
class Trainer:

    def __init__(self, config: dict):
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

    def get_model_params(self, model):
        num_model_params = 0
        for param in model.parameters():
            num_model_params += param.flatten().shape[0]
        
        print('Total params in the world model : ', num_model_params)

    def scale_policy_actions(self, policy_actions):
        return (policy_actions*self.scale) + self.offset

    def insert_into_replay_buffer(self, state_, torques, action, termination):
        # obs -> combine state and torques
        # scale the raw policy actions
        # normalize the state and action
        # obs = torch.cat((state_[..., :67], torques), dim=-1)
        obs = state_
        action = self.scale_policy_actions(action)
        # obs_norm, action_norm = z_norm(obs, action, self.state_mean, self.state_std, self.action_mean, self.action_std)
        state_action_pair = torch.concat([obs, action], dim=-1)
        obs_norm, action_norm = minmax_norm_state_action_pair(state_action_pair, 
                                                              self.jmin,
                                                              self.jmax,
                                                              self.tau_min,
                                                              self.tau_max)
        
        # print("\n\n")
        
        self.replay_buffer.insert([obs_norm.to(self.config["device"]), 
                                   action_norm.to(self.config["device"]), 
                                   torch.unsqueeze(termination.to(self.config["device"]), dim=-1)])



    def on_the_fly_update(self, itr):
        print("inside on the fly update")
        if self._initialized_ is None:
            # params
            self.M, self.N = self.config['world_model_training_params']['M'], self.config['world_model_training_params']['N']
            self.decay = self.config['world_model_training_params']['forecast_decay']
            self.state_dims = self.config['robot_params']['state_dims']
            self.action_dims = self.config['robot_params']['action_dims']
            self.num_mini_batches = self.config["world_model_training_params"]["num_mini_batches"]
            self.total_epochs = self.config['world_model_training_params']['epochs']
            self.total_grad_steps = self.num_mini_batches * self.total_epochs 

            # create model instance
            self.world_model = CreateWorlModelInstance(self.config)
            self.world_model.train()
            print('World Model instantiated\n\n\n')
            count_parameters(self.world_model)

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
                    st_next_pred, ht, std_logits, state_mean = self.world_model.forward(torch.unsqueeze(x[:, t, :], dim=1), ht, predict=True, x_prev=x_prev)
                else:
                    st_next_pred, ht, std_logits, state_mean = self.world_model.forward(torch.cat((st_next_pred, torch.unsqueeze(x[:, t, -self.action_dims:], dim=1)), dim=2), 
                                                                            ht, predict=True, x_prev=st_next_pred)
                target = torch.unsqueeze(x[:, t+1, :self.state_dims], dim=1)
                # loss_t = world_model.nll_loss(dist, target)
                # print(st_next_pred.shape)
                # print(target.shape)
                # print("\n")
                # loss_t = self.world_model.mse_loss(st_pred=torch.squeeze(st_next_pred, dim=1),
                #                                   st_true=torch.squeeze(target, dim=1))
                loss_t = self.world_model.gnll_loss(state_mean=state_mean,
                                               state_std=std_logits,
                                               state_target=target)
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


    # def update(self, model_type, load_dataset):
        
    #     # get data loader obj
    #     wt = self.config['world_model_training_params']
    #     db_paths = self.config.get('db_paths') or [self.config['db_path']]
    #     self.data_loader = load_dataset(
    #         db_paths=db_paths,
    #         batch_size=wt['batch_size'],
    #         M=wt['M'],
    #         N=wt['N'],
    #         combined_db_path=self.config.get("combined_db_path"),
    #         run_mode=self.config.get("run_mode"),
    #         mean=self.config.get("mean_state_action"),
    #         std=self.config.get("std_state_action")
    #     )

    #     # create model instance
    #     M, N = self.config['world_model_training_params']['M'], self.config['world_model_training_params']['N']
    #     decay = self.config['world_model_training_params']['forecast_decay']
    #     state_dims = self.config['robot_params']['state_dims']
    #     action_dims = self.config['robot_params']['action_dims']
    #     world_model = CreateWorlModelInstance(self.config)
    #     world_model.train()
    #     print('World Model instantiated')
    #     # self.get_model_params(world_model)
    #     count_parameters(world_model)
    #     training_loss = []
    #     # model dir 
    #     model_dir_name = get_model_name(model_type)
    #     print('this is the model name : ', model_dir_name)


    #     # load checkpoints
    #     if self.config["use_ckpt"]:
    #         world_model = LoadCkpt(world_model,
    #                                self.config["ckpt_name"],
    #                                self.config["ckpt_dir"]) 



    #     ######################### Training Starts #########################

    #     # Iterate over epochs
    #     for epoch in range(1, self.config['world_model_training_params']['epochs']+1):
    #         print(f'start of epoch {epoch}')
    #         # Iterate over batches
    #         epoch_loss = 0.0
    #         for step, batch_st_ct_at in enumerate(self.data_loader):
    #             ht = torch.zeros((self.config['world_model_arch_params']['num_gru_layers'], 
    #                               batch_st_ct_at.shape[0], 
    #                               self.config['world_model_arch_params']['gru_hidden_dim'])).to(self.config['device'])
                
    #             x = batch_st_ct_at.to(self.config["device"]) # x -> (bs, M+N, s_dim+a_dim)
    #             seq_len = x.shape[1]
    #             batch_loss = 0
    #             alpha = 1.0
    #             # Iterate over RNN timestamps
    #             for t in range(seq_len-1):
    #                 loss_t = 0
    #                 if t < M-1:
    #                     ht = world_model.forward(torch.unsqueeze(x[:, t, :], dim=1), 
    #                                              ht, predict=False) # torch.unsqueeze(x[:, t, :], dim=1) => (bs, s_dim+a_dim) -> (bs, 1, s_dim+a_dim)
    #                 else:
    #                     if t == M-1:
    #                         x_prev = torch.unsqueeze(x[:, t, :96], dim=1)
    #                         # std_logits, state_mean
    #                         st_next_pred, ht = world_model.forward(torch.unsqueeze(x[:, t, :], dim=1), ht, predict=True, x_prev=x_prev)
    #                     else:
    #                         st_next_pred, ht = world_model.forward(torch.cat((st_next_pred, torch.unsqueeze(x[:, t, -action_dims:], dim=1)), dim=2), 
    #                                                                              ht, predict=True, x_prev=st_next_pred)
    #                     target = torch.unsqueeze(x[:, t+1, :state_dims], dim=1)
    #                     # loss_t = world_model.nll_loss(dist, target)
    #                     # print(st_next_pred.shape)
    #                     # print(target.shape)
    #                     # print("\n")
    #                     loss_t = world_model.mse_loss(st_pred=torch.squeeze(st_next_pred, dim=1),
    #                                                   st_true=torch.squeeze(target, dim=1))
    #                     # loss_t = world_model.gnll_loss(state_mean=state_mean,
    #                     #                                state_std=std_logits,
    #                     #                                state_target=target)
    #                     batch_loss += alpha * loss_t
    #                     alpha *= decay
                
    #             batch_loss /= N

    #             # optimize
    #             world_model.optimizer.zero_grad()
    #             batch_loss.backward()
    #             world_model.optimizer.step()

    #             epoch_loss += batch_loss
    #             training_loss.append(batch_loss.item())
    #             # print(f'Loss at step {step} : {batch_loss.item()}')


    #         print(f'end of epoch {epoch}\n\n')
            
    #         # save model
    #         if epoch % self.config["model_save_freq"] == 0:
    #             model_name = f'{model_type}-epoch_{epoch}.pth'
    #             SaveModel(world_model, model_name, model_dir_name)

    #         # save checkpoint
    #         if epoch % self.config["ckpt_save_freq"] == 0:
    #             model_name = f"{model_type}-ckpt-epoch_{epoch}.pth"
    #             SaveCkpt(world_model, model_name, model_dir_name, epoch, epoch_loss)


        
    #     ######################### Training Ends #########################


    