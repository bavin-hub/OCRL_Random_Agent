import torch
import numpy as np
import os
from torch import nn
import torch.optim as optim
from torch.distributions.normal import Normal
from typing import Tuple



# # WM without uncertainity (naive)
# class RandomWorldStepGru(nn.Module):
#     def __init__(self, batch_size: int, state_dim: int, contact_dim: int, action_dim: int, embed_dim: int, 
#                        hidden_dim: int, num_gru_layers: int, mlp_dim: int, lr: float, weight_decay: float, 
#                        device: str, b: int = 0, std_range: Tuple = (0.03, 5), std_init: float = 0.4):
#         super(RandomWorldStepGru, self).__init__()
#         self.lr = lr
#         # self.ln = nn.LayerNorm(state_dim + action_dim)
#         # self.state_action_encoder = nn.Linear(state_dim + action_dim, embed_dim)
#         self.gru_unit = nn.GRU(input_size=state_dim + action_dim, hidden_size=hidden_dim, batch_first=True, num_layers=num_gru_layers)
#         self.state_mlp_head = nn.Linear(hidden_dim, mlp_dim)
#         self.state_mlp_act = nn.ReLU()
#         self.mu_head = nn.Linear(mlp_dim, state_dim)
#         self.mu_act = nn.Tanh()
#         self.logstd_head = nn.Linear(mlp_dim, state_dim)
        
        
#         # logstd
#         self.logstd_range = (np.log(std_range[0]), np.log(std_range[1]))
#         # self.logstd_net = nn.Parameter(torch.full((1, state_dim), np.log(std_init), dtype=torch.float32))
        
#         self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
#         self.state_discrepancy = nn.MSELoss()

#         self.to(device if torch.cuda.is_available() else 'cpu')
    

#     def nll_loss(self, dist, target):
#         loss = -dist.log_prob(target).sum(dim=-1).mean()
#         # print(loss.shape)
#         return loss


#     def forward(self, input_st_at, hidden_state, predict: bool = False, sample: bool = False):
        
#         # ln_out = self.ln(input_st_at)

#         # encoder_out = self.state_action_encoder(input_st_at)
#         gru_out, hidden_state = self.gru_unit(input_st_at, hidden_state)

#         if predict:
#             head_out = self.state_mlp_act(self.state_mlp_head(gru_out))
#             mu_t_next = self.mu_act(self.mu_head(head_out))
#             # mu_t_next = self.mu_head(head_out)
#             logstd_t_next = self.logstd_head(head_out).clamp(*self.logstd_range)

#             if sample:
#                 return mu_t_next, hidden_state

#             # reparameterization trick
#             # self.logstd_net.data.clip(*self.logstd_range)
#             # std_logits = self.logstd_net.exp().expand_as(mu_t_next)
#             std_logits = logstd_t_next.exp()
#             dist = Normal(mu_t_next, std_logits)

#             # logstd = self.logstd_net.clamp(*self.logstd_range)
#             # std_t_next = logstd.exp().expand_as(mu_t_next)
#             # probs = Normal(mu_t_next, std_t_next)

#             st_next_pred = dist.rsample()
            
#             # return mu_t_next, hidden_state
#             return (st_next_pred, hidden_state, dist)
        
#         return (hidden_state)




# # WM without uncertainity (with only residual connection)
# class RandomWorldStepGru(nn.Module):
#     def __init__(self, batch_size: int, state_dim: int, contact_dim: int, action_dim: int, embed_dim: int, 
#                        hidden_dim: int, num_gru_layers: int, mlp_dim: int, lr: float, weight_decay: float, 
#                        device: str, b: int = 0, std_range: Tuple = (0.03, 5), std_init: float = 0.4):
#         super(RandomWorldStepGru, self).__init__()
#         self.lr = lr
#         # self.ln = nn.LayerNorm(state_dim + action_dim)
#         # self.state_action_encoder = nn.Linear(state_dim + action_dim, embed_dim)
#         self.gru_unit = nn.GRU(input_size=state_dim + action_dim, hidden_size=hidden_dim, batch_first=True, num_layers=num_gru_layers)
#         self.state_mlp_head = nn.Linear(hidden_dim, mlp_dim)
#         self.state_mlp_act = nn.ReLU()
#         self.mu_head = nn.Linear(mlp_dim, state_dim)
#         # self.mu_act = nn.Tanh()
#         self.logstd_head = nn.Linear(mlp_dim, state_dim)
        
        
#         # logstd
#         self.logstd_range = (np.log(std_range[0]), np.log(std_range[1]))
#         # self.logstd_net = nn.Parameter(torch.full((1, state_dim), np.log(std_init), dtype=torch.float32))
        
#         self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
#         self.state_discrepancy = nn.MSELoss()

#         self.to(device if torch.cuda.is_available() else 'cpu')
    

#     def nll_loss(self, dist, target):
#         loss = -dist.log_prob(target).sum(dim=-1).mean()
#         # print(loss.shape)
#         return loss


#     def forward(self, input_st_at, hidden_state, predict: bool = False, sample: bool = False, x_prev=None):
        
#         # ln_out = self.ln(input_st_at)

#         # encoder_out = self.state_action_encoder(input_st_at)
#         gru_out, hidden_state = self.gru_unit(input_st_at, hidden_state)

#         if predict:
#             head_out = self.state_mlp_act(self.state_mlp_head(gru_out))
#             mu_t_next = self.mu_head(head_out) + x_prev
#             # print(mu_t_next.shape)
#             # print('this is x_pred : ', x_prev.shape)
#             # print('after adding : ', (mu_t_next + x_prev).shape)
#             # print("\n")
#             # mu_t_next = self.mu_head(head_out)
#             logstd_t_next = self.logstd_head(head_out).clamp(*self.logstd_range)

#             if sample:
#                 return mu_t_next, hidden_state

#             # reparameterization trick
#             # self.logstd_net.data.clip(*self.logstd_range)
#             # std_logits = self.logstd_net.exp().expand_as(mu_t_next)
#             std_logits = logstd_t_next.exp()
#             dist = Normal(mu_t_next, std_logits)

#             # logstd = self.logstd_net.clamp(*self.logstd_range)
#             # std_t_next = logstd.exp().expand_as(mu_t_next)
#             # probs = Normal(mu_t_next, std_t_next)

#             st_next_pred = dist.rsample()
            
#             # return mu_t_next, hidden_state
#             return (st_next_pred, hidden_state, dist)
        
#         return (hidden_state)
    



# # WM without uncertainity (with residual connection and 2 separate mlps for mu and logstd)
# class RandomWorldStepGru(nn.Module):
#     def __init__(self, batch_size: int, state_dim: int, contact_dim: int, action_dim: int, embed_dim: int, 
#                        hidden_dim: int, num_gru_layers: int, mlp_dim: int, lr: float, weight_decay: float, 
#                        device: str, b: int = 0, std_range: Tuple = (0.03, 5), std_init: float = 0.4):
#         super(RandomWorldStepGru, self).__init__()
#         self.device = device
#         self.lr = lr
#         # self.ln = nn.LayerNorm(state_dim + action_dim)
#         # self.state_action_encoder = nn.Linear(state_dim + action_dim, embed_dim)
#         self.gru_unit = nn.GRU(input_size=state_dim + action_dim, hidden_size=hidden_dim, batch_first=True, num_layers=num_gru_layers)
        
#         # mlp for mu
#         self.mean_mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_dim),
#                                       nn.ReLU(),
#                                       nn.Linear(mlp_dim, state_dim))
        
#         # mlp for logstd
#         self.logstd_mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_dim),
#                                         nn.ReLU(),
#                                         nn.Linear(mlp_dim, state_dim))
        
#         # self.state_mlp_head = nn.Linear(hidden_dim, mlp_dim)
#         # self.state_mlp_act = nn.ReLU()
#         # self.mu_head = nn.Linear(mlp_dim, state_dim)
#         # # self.mu_act = nn.Tanh()
#         # self.logstd_head = nn.Linear(mlp_dim, state_dim)
        
        
#         # logstd init
#         self.logstd_range = (np.log(std_range[0]), np.log(std_range[1]))
#         self.state_min_logstd = nn.Parameter(torch.ones(1, state_dim, device=device) * -5.0)
#         self.state_log_delta_logstd = nn.Parameter(torch.ones(1, state_dim, device=device) * 0.0)
#         # self.logstd_net = nn.Parameter(torch.full((1, state_dim), np.log(std_init), dtype=torch.float32))
        
#         self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
#         self.state_discrepancy = nn.MSELoss()

#         self.to(device if torch.cuda.is_available() else 'cpu')
    

#     def nll_loss(self, dist, target):
#         loss = -dist.log_prob(target).sum(dim=-1).mean()
#         # print(loss.shape)
#         return loss
    
#     def mse_loss(self, st_true, st_pred):
#         loss = self.state_discrepancy(st_pred, st_true)
#         return loss


#     def forward(self, input_st_at, hidden_state, predict: bool = False, sample: bool = False, x_prev=None):
        
#         # ln_out = self.ln(input_st_at)

#         # encoder_out = self.state_action_encoder(input_st_at)
#         gru_out, hidden_state = self.gru_unit(input_st_at, hidden_state)

#         if predict:
#             # residual skip connection
#             mu_t_next = self.mean_mlp(gru_out) + x_prev
            
#             # mu_t_next = self.mu_head(head_out) + x_prev
#             # print(mu_t_next.shape)
#             # print('this is x_pred : ', x_prev.shape)
#             # print('after adding : ', (mu_t_next + x_prev).shape)
#             # print("\n")
#             # mu_t_next = self.mu_head(head_out)
#             # logstd_t_next = self.logstd_head(head_out).clamp(*self.logstd_range)
            
#             logstd_t_next = self.logstd_mlp(gru_out).clamp(*self.logstd_range)

#             # clipping the logstd_t_next
#             # state_max_logstd = self.state_min_logstd + torch.exp(self.state_log_delta_logstd)
#             # logstd_t_next = state_max_logstd - nn.functional.softplus(state_max_logstd - logstd_t_next)
#             # logstd_t_next = self.state_min_logstd + nn.functional.softplus(logstd_t_next - self.state_min_logstd)

#             # making deterministic predictions
#             if sample:
#                 return mu_t_next, hidden_state

#             # reparameterization trick
#             # self.logstd_net.data.clip(*self.logstd_range)
#             # std_logits = self.logstd_net.exp().expand_as(mu_t_next)
#             std_logits = logstd_t_next.exp()
#             dist = Normal(mu_t_next, std_logits)
#             st_next_pred = dist.rsample()

#             # st_next_pred = (torch.randn_like(mu_t_next, device=self.device) * std_logits + mu_t_next)


#             # logstd = self.logstd_net.clamp(*self.logstd_range)
#             # std_t_next = logstd.exp().expand_as(mu_t_next)
#             # probs = Normal(mu_t_next, std_t_next)

            
            
#             # return mu_t_next, hidden_state
#             return (st_next_pred, hidden_state, dist)
        
#         return (hidden_state)






# # WM without uncertainity (with residual connection and 2 separate mlps for mu and logstd, mse loss)
# class RandomWorldStepGru(nn.Module):
#     def __init__(self, batch_size: int, state_dim: int, contact_dim: int, action_dim: int, embed_dim: int, 
#                        hidden_dim: int, num_gru_layers: int, mlp_dim: int, lr: float, weight_decay: float, 
#                        device: str, b: int = 0, std_range: Tuple = (0.03, 5), std_init: float = 0.4):
#         super(RandomWorldStepGru, self).__init__()
#         self.device = device
#         self.lr = lr
#         # self.ln = nn.LayerNorm(state_dim + action_dim)
#         # self.state_action_encoder = nn.Linear(state_dim + action_dim, embed_dim)
#         self.gru_unit = nn.GRU(input_size=state_dim + action_dim, hidden_size=hidden_dim, batch_first=True, num_layers=num_gru_layers)
        
#         # mlp for mu
#         self.mean_mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_dim),
#                                       nn.ReLU(),
#                                       nn.Linear(mlp_dim, state_dim))
        
#         # mlp for logstd
#         self.logstd_mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_dim),
#                                         nn.ReLU(),
#                                         nn.Linear(mlp_dim, state_dim))
        
#         # self.state_mlp_head = nn.Linear(hidden_dim, mlp_dim)
#         # self.state_mlp_act = nn.ReLU()
#         # self.mu_head = nn.Linear(mlp_dim, state_dim)
#         # # self.mu_act = nn.Tanh()
#         # self.logstd_head = nn.Linear(mlp_dim, state_dim)
        
        
#         # logstd init
#         self.logstd_range = (np.log(std_range[0]), np.log(std_range[1]))
#         self.state_min_logstd = nn.Parameter(torch.ones(1, state_dim, device=device) * -10.0)
#         self.state_log_delta_logstd = nn.Parameter(torch.ones(1, state_dim, device=device) * 0.0)
#         # self.logstd_net = nn.Parameter(torch.full((1, state_dim), np.log(std_init), dtype=torch.float32))
        
#         self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
#         self.state_discrepancy = nn.MSELoss()

#         self.to(device if torch.cuda.is_available() else 'cpu')
    

#     def nll_loss(self, dist, target):
#         loss = -dist.log_prob(target).sum(dim=-1).mean()
#         # print(loss.shape)
#         return loss
    
#     def mse_loss(self, st_pred, st_true):
#         # print(torch.squeeze(st_pred, dim=1).shape)
#         loss = torch.sum(torch.square(st_pred - st_true), dim=1).mean(dim=0)
#         # print(loss.shape)
#         return loss


#     def forward(self, input_st_at, hidden_state, predict: bool = False, sample: bool = False, x_prev=None):

#         # encoder_out = self.state_action_encoder(input_st_at)
#         gru_out, hidden_state = self.gru_unit(input_st_at, hidden_state)

#         if predict:
#             # residual skip connection
#             mu_t_next = self.mean_mlp(gru_out) + x_prev
            
#             # logstd prediction
#             logstd_t_next = self.logstd_mlp(gru_out).clamp(*self.logstd_range)

#             # clipping the logstd_t_next
#             # state_max_logstd = self.state_min_logstd + torch.exp(self.state_log_delta_logstd)
#             # logstd_t_next = state_max_logstd - nn.functional.softplus(state_max_logstd - logstd_t_next)
#             # logstd_t_next = self.state_min_logstd + nn.functional.softplus(logstd_t_next - self.state_min_logstd)
#             std_logits = logstd_t_next.exp()

#             # making deterministic predictions
#             if sample:
#                 return mu_t_next, hidden_state

#             # reparameterization trick
#             st_next_pred = torch.randn_like(mu_t_next, device=self.device) * std_logits + mu_t_next
#             # noise = torch.randn_like(mu_t_next).detach()
#             # st_next_pred = (noise * std_logits + mu_t_next)
            
            
#             # return mu_t_next, hidden_state
#             return (st_next_pred, hidden_state)
        
#         return (hidden_state)






# WM without uncertainity (with residual connection and 2 separate mlps for mu and logstd, mse loss)
class RandomWorldStepGru(nn.Module):
    def __init__(self, batch_size: int, state_dim: int, contact_dim: int, action_dim: int, embed_dim: int, 
                       hidden_dim: int, num_gru_layers: int, mlp_dim: int, lr: float, weight_decay: float, 
                       device: str, b: int = 0, std_range: Tuple = (0.03, 5), std_init: float = 0.4):
        super(RandomWorldStepGru, self).__init__()
        self.device = device
        self.lr = lr
        # self.ln = nn.LayerNorm(state_dim + action_dim)
        # self.state_action_encoder = nn.Linear(state_dim + action_dim, embed_dim)
        self.gru_unit = nn.GRU(input_size=state_dim + action_dim, hidden_size=hidden_dim, batch_first=True, num_layers=num_gru_layers)
        
        # mlp for mu
        self.mean_mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_dim),
                                      nn.ReLU(),
                                      nn.Linear(mlp_dim, state_dim))
        
        # mlp for logstd
        self.logstd_mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_dim),
                                        nn.ReLU(),
                                        nn.Linear(mlp_dim, state_dim))
        
        # self.state_mlp_head = nn.Linear(hidden_dim, mlp_dim)
        # self.state_mlp_act = nn.ReLU()
        # self.mu_head = nn.Linear(mlp_dim, state_dim)
        # # self.mu_act = nn.Tanh()
        # self.logstd_head = nn.Linear(mlp_dim, state_dim)
        
        
        # logstd init
        self.logstd_range = (np.log(std_range[0]), np.log(std_range[1]))
        self.state_min_logstd = nn.Parameter(torch.ones(1, state_dim, device=device) * -3.5)
        self.state_log_delta_logstd = nn.Parameter(torch.ones(1, state_dim, device=device) * 1.0)
        # self.logstd_net = nn.Parameter(torch.full((1, state_dim), np.log(std_init), dtype=torch.float32))
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
        # self.state_discrepancy = nn.MSELoss()
        self.state_discrepancy = nn.GaussianNLLLoss()

        self.to(device if torch.cuda.is_available() else 'cpu')
    
    def gnll_loss(self, state_mean, state_std, state_target):
        state_loss = self.state_discrepancy(state_mean, state_target, state_std ** 2)
        std_loss = torch.mean(self.state_max_logstd) - torch.mean(self.state_min_logstd)
        return state_loss + std_loss

    def nll_loss(self, dist, target):
        loss = -dist.log_prob(target).sum(dim=-1).mean()
        # print(loss.shape)
        return loss
    
    def mse_loss(self, st_pred, st_true):
        # print(torch.squeeze(st_pred, dim=1).shape)
        loss = torch.sum(torch.square(st_pred - st_true), dim=1).mean(dim=0)
        # print(loss.shape)
        return loss


    def forward(self, input_st_at, hidden_state, predict: bool = False, sample: bool = False, x_prev=None):

        # encoder_out = self.state_action_encoder(input_st_at)
        gru_out, hidden_state = self.gru_unit(input_st_at, hidden_state)

        if predict:
            # residual skip connection
            mu_t_next = self.mean_mlp(gru_out) + x_prev
            
            # logstd prediction
            logstd_t_net_out = self.logstd_mlp(gru_out)
            # logstd_t_next = logstd_t_net_out.clamp(*self.logstd_range)
            # clipping the logstd_t_next
            self.state_max_logstd = self.state_min_logstd + torch.exp(self.state_log_delta_logstd)
            logstd_t_next = self.state_max_logstd - nn.functional.softplus(self.state_max_logstd - logstd_t_net_out)
            logstd_t_next = self.state_min_logstd + nn.functional.softplus(logstd_t_next - self.state_min_logstd)
            # .clamp(-5.0, 2.0)
            std_logits = logstd_t_next.exp()

            # making deterministic predictions
            if sample:
                return mu_t_next, hidden_state

            # reparameterization trick
            st_next_pred = torch.randn_like(mu_t_next, device=self.device) * std_logits + mu_t_next
            # noise = torch.randn_like(mu_t_next).detach()
            # st_next_pred = (noise * std_logits + mu_t_next)

            # print("logstd network out : ", logstd_t_net_out, "  logstd clampled : ", logstd_t_next, " std out : ", std_logits)
            # print("logstd clampled : ", logstd_t_next[0, :])
            # print("\n\n")
            
            
            # return mu_t_next, hidden_state
            return (st_next_pred, hidden_state, std_logits, mu_t_next)
        
        return (hidden_state)



# # WM without uncertainity (with residual connection and 2 separate mlps for mu and logstd, mse loss)
# class RandomWorldStepGru(nn.Module):
#     def __init__(self, batch_size: int, state_dim: int, contact_dim: int, action_dim: int, embed_dim: int, 
#                        hidden_dim: int, num_gru_layers: int, mlp_dim: int, lr: float, weight_decay: float, 
#                        device: str, b: int = 0, std_range: Tuple = (0.03, 5), std_init: float = 0.4):
#         super(RandomWorldStepGru, self).__init__()
#         self.device = device
#         self.lr = lr
#         # self.ln = nn.LayerNorm(state_dim + action_dim)
#         # self.state_action_encoder = nn.Linear(state_dim + action_dim, embed_dim)
#         self.gru_unit = nn.GRU(input_size=state_dim + action_dim, hidden_size=hidden_dim, batch_first=True, num_layers=num_gru_layers)
        
#         # mlp for mu
#         self.mean_mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_dim),
#                                       nn.ReLU(),
#                                       nn.Linear(mlp_dim, state_dim))
        
#         # mlp for logstd
#         self.logstd_mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_dim),
#                                         nn.ReLU(),
#                                         nn.Linear(mlp_dim, state_dim))
        
#         # self.state_mlp_head = nn.Linear(hidden_dim, mlp_dim)
#         # self.state_mlp_act = nn.ReLU()
#         # self.mu_head = nn.Linear(mlp_dim, state_dim)
#         # # self.mu_act = nn.Tanh()
#         # self.logstd_head = nn.Linear(mlp_dim, state_dim)
        
        
#         # logstd init
#         self.logstd_range = (np.log(std_range[0]), np.log(std_range[1]))
#         self.state_min_logstd = nn.Parameter(torch.ones(1, state_dim, device=device) * -10.0)
#         self.state_log_delta_logstd = nn.Parameter(torch.ones(1, state_dim, device=device) * 0.0)
#         # self.logstd_net = nn.Parameter(torch.full((1, state_dim), np.log(std_init), dtype=torch.float32))
        
#         self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
#         self.state_discrepancy = nn.MSELoss()

#         self.to(device if torch.cuda.is_available() else 'cpu')
    

#     def nll_loss(self, dist, target):
#         loss = -dist.log_prob(target).sum(dim=-1).mean()
#         # print(loss.shape)
#         return loss
    
#     def mse_loss(self, st_pred, st_true):
#         # print(torch.squeeze(st_pred, dim=1).shape)
#         loss = torch.sum(torch.square(st_pred - st_true), dim=1).mean(dim=0)
#         # print(loss.shape)
#         return loss


#     def forward(self, input_st_at, hidden_state, predict: bool = False, sample: bool = False, x_prev=None):

#         # encoder_out = self.state_action_encoder(input_st_at)
#         gru_out, hidden_state = self.gru_unit(input_st_at, hidden_state)

#         if predict:
#             # residual skip connection
#             mu_t_next = self.mean_mlp(gru_out) + x_prev
            
#             # logstd prediction
#             # logstd_t_next = self.logstd_mlp(gru_out).clamp(*self.logstd_range)

#             # clipping the logstd_t_next
#             # state_max_logstd = self.state_min_logstd + torch.exp(self.state_log_delta_logstd)
#             # logstd_t_next = state_max_logstd - nn.functional.softplus(state_max_logstd - logstd_t_next)
#             # logstd_t_next = self.state_min_logstd + nn.functional.softplus(logstd_t_next - self.state_min_logstd)
#             # std_logits = logstd_t_next.exp()

#             # making deterministic predictions
#             if sample:
#                 return mu_t_next, hidden_state

#             # reparameterization trick
#             # st_next_pred = torch.randn_like(mu_t_next, device=self.device) * std_logits + mu_t_next
#             # noise = torch.randn_like(mu_t_next).detach()
#             # st_next_pred = (noise * std_logits + mu_t_next)
            
            
#             # return mu_t_next, hidden_state
#             return (mu_t_next, hidden_state)
        
#         return (hidden_state)


    
    





        




