import torch
import numpy as np
import os
from torch import nn
import torch.optim as optim
from torch.distributions.normal import Normal
from typing import Tuple





# # WM without uncertainity (mse; znorm; hard-coded bounds)
# class RandomWorldStepGru(nn.Module):
#     def __init__(self, batch_size: int, state_dim: int, contact_dim: int, action_dim: int, embed_dim: int, 
#                        hidden_dim: int, num_gru_layers: int, mlp_dim: int, lr: float, weight_decay: float, 
#                        device: str, b: int = 0, std_range: Tuple = (0.01, 0.06), std_init: float = 0.4):
#         super(RandomWorldStepGru, self).__init__()

#         # (0.01, 0.06)
#         # (0.23, 0.9)

#         self.device = device
#         self.lr = lr
#         self.gru_unit = nn.GRU(input_size=state_dim + action_dim, hidden_size=hidden_dim, batch_first=True, num_layers=num_gru_layers)
        
#         # mlp for mu
#         self.mean_mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_dim),
#                                       nn.ReLU(),
#                                       nn.Linear(mlp_dim, state_dim))
        
#         # mlp for logstd
#         self.logstd_mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_dim),
#                                         nn.ReLU(),
#                                         nn.Linear(mlp_dim, state_dim))
        
 
        
#         # # logstd init
#         self.logstd_range = (np.log(std_range[0]), np.log(std_range[1]))
#         print("Logstd range : ", self.logstd_range)
#         # self.state_min_logstd = nn.Parameter(torch.ones(1, state_dim, device=device) * -3.5)
#         # self.state_log_delta_logstd = nn.Parameter(torch.ones(1, state_dim, device=device) * 1.0)

#         self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
#         # self.state_discrepancy = nn.MSELoss()
#         self.state_discrepancy = nn.GaussianNLLLoss()

#         self.to(device if torch.cuda.is_available() else 'cpu')
    
#     def gnll_loss(self, state_mean, state_std, state_target):
#         state_loss = self.state_discrepancy(state_mean, state_target, state_std ** 2)
#         std_loss = torch.mean(self.state_max_logstd) - torch.mean(self.state_min_logstd)
#         return state_loss + std_loss

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

#         gru_out, hidden_state = self.gru_unit(input_st_at, hidden_state)

#         if predict:
#             # residual skip connection
#             mu_t_next = self.mean_mlp(gru_out) + x_prev
            
#             # # logstd prediction
#             logstd_t_net_out = self.logstd_mlp(gru_out).clamp(*self.logstd_range)
#             # # logstd_t_next = logstd_t_net_out.clamp(*self.logstd_range)
#             # # clipping the logstd_t_next
#             # self.state_max_logstd = self.state_min_logstd + torch.exp(self.state_log_delta_logstd)
#             # logstd_t_next = self.state_max_logstd - nn.functional.softplus(self.state_max_logstd - logstd_t_net_out)
#             # logstd_t_next = self.state_min_logstd + nn.functional.softplus(logstd_t_next - self.state_min_logstd)
#             # # .clamp(-5.0, 2.0)
#             std_logits = logstd_t_net_out.exp()

#             # making deterministic predictions
#             if sample:
#                 return mu_t_next, hidden_state

#             # reparameterization trick
#             st_next_pred = torch.randn_like(mu_t_next, device=self.device) * std_logits + mu_t_next
#             # st_next_pred = torch.randn_like(mu_t_next, device=self.device) + mu_t_next

            
            
#             # return mu_t_next, hidden_state
#             return (st_next_pred, hidden_state)
        
#         return (hidden_state)





# WM without uncertainity (with residual connection and 2 separate mlps for mu and logstd, mse loss)
class RandomWorldStepGru(nn.Module):
    def __init__(self, batch_size: int, state_dim: int, contact_dim: int, action_dim: int, embed_dim: int, 
                       hidden_dim: int, num_gru_layers: int, mlp_dim: int, lr: float, weight_decay: float, 
                       device: str, b: int = 0, std_range: Tuple = (0.03,5), std_init: float = 0.4):
        super(RandomWorldStepGru, self).__init__()

        # (0.01, 0.06)
        # (0.23, 0.9)
        # (0.03,5)

        self.device = device
        self.lr = lr
        self.gru_unit = nn.GRU(input_size=state_dim + action_dim, hidden_size=hidden_dim, batch_first=True, num_layers=num_gru_layers)
        
        # self.final_act = nn.Tanh()
        
        # mlp for mu
        self.mean_mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_dim),
                                      nn.ReLU(),
                                      nn.Linear(mlp_dim, state_dim))
        
        # mlp for logstd
        self.logstd_mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_dim),
                                        nn.ReLU(),
                                        nn.Linear(mlp_dim, state_dim))
        
 
        
        # # logstd init
        self.logstd_range = (np.log(std_range[0]), np.log(std_range[1]))
        print("Logstd range : ", self.logstd_range)
        # self.state_min_logstd = nn.Parameter(torch.ones(1, state_dim, device=device) * -3.5)
        # self.state_log_delta_logstd = nn.Parameter(torch.ones(1, state_dim, device=device) * 1.0)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
        # self.state_discrepancy = nn.MSELoss()
        self.state_discrepancy = nn.GaussianNLLLoss()

        self.to(device if torch.cuda.is_available() else 'cpu')
    
    def gnll_loss(self, state_mean, state_std, state_target):
        state_loss = self.state_discrepancy(state_mean, state_target, state_std ** 2)
        # std_loss = torch.mean(self.state_max_logstd) - torch.mean(self.state_min_logstd)
        return state_loss 

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

        gru_out, hidden_state = self.gru_unit(input_st_at, hidden_state)

        if predict:
            # residual skip connection
            mu_t_next = self.mean_mlp(gru_out) + x_prev
            
            # # logstd prediction
            logstd_t_net_out = self.logstd_mlp(gru_out).clamp(*self.logstd_range)
            # # logstd_t_next = logstd_t_net_out.clamp(*self.logstd_range)
            # # clipping the logstd_t_next
            # self.state_max_logstd = self.state_min_logstd + torch.exp(self.state_log_delta_logstd)
            # logstd_t_next = self.state_max_logstd - nn.functional.softplus(self.state_max_logstd - logstd_t_net_out)
            # logstd_t_next = self.state_min_logstd + nn.functional.softplus(logstd_t_next - self.state_min_logstd)
            # # .clamp(-5.0, 2.0)
            std_logits = logstd_t_net_out.exp()

            # making deterministic predictions
            if sample:
                return mu_t_next, hidden_state

            # reparameterization trick
            st_next_pred = torch.randn_like(mu_t_next, device=self.device) * std_logits + mu_t_next
            # st_next_pred = torch.randn_like(mu_t_next, device=self.device) + mu_t_next

            
            
            # return mu_t_next, hidden_state
            return (st_next_pred, hidden_state, std_logits, mu_t_next)
        
        return (hidden_state)












# # WM without uncertainity (just mse)
# class RandomWorldStepGru(nn.Module):
#     def __init__(self, batch_size: int, state_dim: int, contact_dim: int, action_dim: int, embed_dim: int, 
#                        hidden_dim: int, num_gru_layers: int, mlp_dim: int, lr: float, weight_decay: float, 
#                        device: str, b: int = 0, std_range: Tuple = (0.03,5), std_init: float = 0.4):
#         super(RandomWorldStepGru, self).__init__()



#         self.device = device
#         self.lr = lr
#         self.gru_unit = nn.GRU(input_size=state_dim + action_dim, hidden_size=hidden_dim, batch_first=True, num_layers=num_gru_layers)
        
#         # mlp for mu
#         self.mean_mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_dim),
#                                       nn.ReLU(),
#                                       nn.Linear(mlp_dim, state_dim))
        
#         # # mlp for logstd
#         # self.logstd_mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_dim),
#         #                                 nn.ReLU(),
#         #                                 nn.Linear(mlp_dim, state_dim))
        
 
        
#         # # # logstd init
#         # self.logstd_range = (np.log(std_range[0]), np.log(std_range[1]))
#         # print("Logstd range : ", self.logstd_range)
#         # # self.state_min_logstd = nn.Parameter(torch.ones(1, state_dim, device=device) * -3.5)
#         # # self.state_log_delta_logstd = nn.Parameter(torch.ones(1, state_dim, device=device) * 1.0)

#         self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
#         self.state_discrepancy = nn.MSELoss()
#         # self.state_discrepancy = nn.GaussianNLLLoss()

#         self.to(device if torch.cuda.is_available() else 'cpu')
    
#     def gnll_loss(self, state_mean, state_std, state_target):
#         state_loss = self.state_discrepancy(state_mean, state_target, state_std ** 2)
#         # std_loss = torch.mean(self.state_max_logstd) - torch.mean(self.state_min_logstd)
#         return state_loss 

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

#         gru_out, hidden_state = self.gru_unit(input_st_at, hidden_state)

#         if predict:
#             # residual skip connection
#             mu_t_next = self.mean_mlp(gru_out) + x_prev
            
#             # # # logstd prediction
#             # logstd_t_net_out = self.logstd_mlp(gru_out).clamp(*self.logstd_range)
#             # # # logstd_t_next = logstd_t_net_out.clamp(*self.logstd_range)
#             # # # clipping the logstd_t_next
#             # # self.state_max_logstd = self.state_min_logstd + torch.exp(self.state_log_delta_logstd)
#             # # logstd_t_next = self.state_max_logstd - nn.functional.softplus(self.state_max_logstd - logstd_t_net_out)
#             # # logstd_t_next = self.state_min_logstd + nn.functional.softplus(logstd_t_next - self.state_min_logstd)
#             # # # .clamp(-5.0, 2.0)
#             # std_logits = logstd_t_net_out.exp()

#             # making deterministic predictions
#             if sample:
#                 return mu_t_next, hidden_state

#             # reparameterization trick
#             # st_next_pred = torch.randn_like(mu_t_next, device=self.device) * std_logits + mu_t_next
#             st_next_pred = mu_t_next
#             # st_next_pred = torch.randn_like(mu_t_next, device=self.device) + mu_t_next

            
            
#             # return mu_t_next, hidden_state
#             return (st_next_pred, hidden_state)
        
#         return (hidden_state)


    
    





        




