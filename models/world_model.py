import torch
import numpy as np
import os
from torch import nn
import torch.optim as optim
from torch.distributions.normal import Normal
from typing import Tuple



# WM without uncertainity
class RandomWorldStepGru(nn.Module):
    def __init__(self, batch_size: int, state_dim: int, contact_dim: int, action_dim: int, embed_dim: int, 
                       hidden_dim: int, num_gru_layers: int, mlp_dim: int, lr: float, weight_decay: float, 
                       device: str, b: int = 0, std_range: Tuple = (0.03, 5), std_init: float = 0.4):
        super(RandomWorldStepGru, self).__init__()
        self.lr = lr
        # self.ln = nn.LayerNorm(state_dim + action_dim)
        self.state_action_encoder = nn.Linear(state_dim + action_dim, embed_dim)
        self.gru_unit = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True, num_layers=num_gru_layers)
        self.state_mlp_head = nn.Linear(hidden_dim, mlp_dim)
        self.mu_head = nn.Linear(mlp_dim, state_dim)
        self.mu_act = nn.Tanh()
        self.logstd_head = nn.Linear(mlp_dim, state_dim)
        
        
        # logstd
        self.logstd_range = (np.log(std_range[0]), np.log(std_range[1]))
        # self.logstd_net = nn.Parameter(torch.full((1, state_dim), np.log(std_init), dtype=torch.float32))
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.state_discrepancy = nn.MSELoss()

        self.to(device if torch.cuda.is_available() else 'cpu')
    

    def nll_loss(self, dist, target):
        loss = -dist.log_prob(target).sum(dim=-1).mean()
        # print(loss.shape)
        return loss


    def forward(self, input_st_at, hidden_state, predict: bool = False, sample: bool = False):
        
        # ln_out = self.ln(input_st_at)

        encoder_out = self.state_action_encoder(input_st_at)
        gru_out, hidden_state = self.gru_unit(encoder_out, hidden_state)

        if predict:
            head_out = self.state_mlp_head(gru_out)
            mu_t_next = self.mu_act(self.mu_head(head_out))
            # mu_t_next = self.mu_head(head_out)
            logstd_t_next = self.logstd_head(head_out).clamp(*self.logstd_range)

            if sample:
                return mu_t_next, hidden_state

            # reparameterization trick
            # self.logstd_net.data.clip(*self.logstd_range)
            # std_logits = self.logstd_net.exp().expand_as(mu_t_next)
            std_logits = logstd_t_next.exp()
            dist = Normal(mu_t_next, std_logits)

            # logstd = self.logstd_net.clamp(*self.logstd_range)
            # std_t_next = logstd.exp().expand_as(mu_t_next)
            # probs = Normal(mu_t_next, std_t_next)

            st_next_pred = dist.rsample()
            
            # return mu_t_next, hidden_state
            return (st_next_pred, hidden_state, dist)
        
        return (hidden_state)
    


    
    





        




