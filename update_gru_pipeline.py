import sys
import re

# 1. Update models/world_model.py
with open('models/world_model.py', 'r') as f:
    text = f.read()

# I will just write the clean RandomWorldStepGru class and append it/replace it.
clean_gru_class = """
class RandomWorldStepGru(nn.Module):
    def __init__(self, batch_size: int, state_dim: int, contact_dim: int, action_dim: int, embed_dim: int, 
                       hidden_dim: int, num_gru_layers: int, mlp_dim: int, lr: float, weight_decay: float, 
                       device: str, b: int = 0, std_range: tuple = (0.03, 5), std_init: float = 0.4):
        super(RandomWorldStepGru, self).__init__()
        self.device = device
        self.lr = lr
        self.with_uncertainty = False # or read from config if passed
        
        self.gru_unit = nn.GRU(input_size=state_dim + action_dim, hidden_size=hidden_dim, batch_first=True, num_layers=num_gru_layers)
        
        self.mean_mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, state_dim))
        self.logstd_mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, state_dim))
        self.contact_mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, contact_dim))
        
        self.logstd_range = (np.log(std_range[0]), np.log(std_range[1]))
        self.state_min_logstd = nn.Parameter(torch.ones(1, state_dim, device=device) * -3.5)
        self.state_log_delta_logstd = nn.Parameter(torch.ones(1, state_dim, device=device) * 1.0)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.state_discrepancy = nn.GaussianNLLLoss()

        d = str(device)
        if d == "cuda" and not torch.cuda.is_available():
            d = "cpu"
        elif d == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            d = "cpu"
        self.device = torch.device(d)
        self.to(self.device)
    
    def gnll_loss(self, state_mean, state_std, state_target):
        state_loss = self.state_discrepancy(state_mean, state_target, state_std ** 2)
        std_loss = torch.mean(self.state_max_logstd) - torch.mean(self.state_min_logstd)
        return state_loss + std_loss

    def mse_loss(self, st_pred, st_true):
        return torch.sum((st_pred - st_true) ** 2, dim=-1).mean()

    def forward(self, s_t, a_t, h_prev=None, x_prev=None):
        input_st_at = torch.cat([s_t, a_t], dim=-1)
        if input_st_at.dim() == 2:
            input_st_at = input_st_at.unsqueeze(1)
            
        gru_out, h_next = self.gru_unit(input_st_at, h_prev)
        gru_out = gru_out.squeeze(1) if gru_out.dim() == 3 else gru_out
        
        mu_t_next = self.mean_mlp(gru_out)
        if x_prev is not None:
            mu_t_next = mu_t_next + x_prev
            
        contact_pred = self.contact_mlp(gru_out)
        
        if self.with_uncertainty:
            logstd_t_net_out = self.logstd_mlp(gru_out)
            self.state_max_logstd = self.state_min_logstd + torch.exp(self.state_log_delta_logstd)
            logstd_t_next = self.state_max_logstd - nn.functional.softplus(self.state_max_logstd - logstd_t_net_out)
            logstd_t_next = self.state_min_logstd + nn.functional.softplus(logstd_t_next - self.state_min_logstd)
            std_logits = logstd_t_next.exp()
            return mu_t_next, contact_pred, std_logits, h_next
        else:
            return mu_t_next, contact_pred, h_next
"""

# Find where the class RandomWorldStepGru is defined and replace it
import re
text = re.sub(r'class RandomWorldStepGru\(nn\.Module\):.*', clean_gru_class, text, flags=re.DOTALL)

with open('models/world_model.py', 'w') as f:
    f.write(text)


# 2. Update train_mlp.py
with open('train_mlp.py', 'r') as f:
    train_text = f.read()

train_text = train_text.replace(
'''                    out = world_model(s_pred, a_t, x_prev=s_pred)
                    target = x[:, t + 1, :S]
                    if world_model.with_uncertainty:
                        mu, contact_pred, std = out
                        loss_t = world_model.gnll_loss(mu, std, target)
                        loss_c = world_model.mse_loss(contact_pred, c_target)
                        # Reparameterization: differentiable sample, teaches noise robustness
                        s_pred = mu + std * torch.randn_like(std)
                    else:
                        mu, contact_pred = out
                        loss_t = world_model.mse_loss(mu, target)
                        loss_c = world_model.mse_loss(contact_pred, c_target)
                        s_pred = mu''',
'''                    out = world_model(s_pred, a_t, h_prev=h_t, x_prev=s_pred) if hasattr(world_model, "gru_unit") else world_model(s_pred, a_t, x_prev=s_pred)
                    if hasattr(world_model, "gru_unit"):
                        if getattr(world_model, "with_uncertainty", False):
                            mu, contact_pred, std, h_t = out
                        else:
                            mu, contact_pred, h_t = out
                    else:
                        if getattr(world_model, "with_uncertainty", False):
                            mu, contact_pred, std = out
                        else:
                            mu, contact_pred = out
                            
                    target = x[:, t + 1, :S]
                    if getattr(world_model, "with_uncertainty", False):
                        loss_t = world_model.gnll_loss(mu, std, target)
                        loss_c = world_model.mse_loss(contact_pred, c_target)
                        s_pred = mu + std * torch.randn_like(std)
                    else:
                        loss_t = world_model.mse_loss(mu, target)
                        loss_c = world_model.mse_loss(contact_pred, c_target)
                        s_pred = mu'''
)

# Initialize h_t = None
train_text = train_text.replace(
'''                s_pred = x[:, 0, :S]  # real first state
                batch_loss = 0.0
                alpha = 1.0

                for t in range(N):''',
'''                s_pred = x[:, 0, :S]  # real first state
                batch_loss = 0.0
                alpha = 1.0
                h_t = None

                for t in range(N):'''
)

with open('train_mlp.py', 'w') as f:
    f.write(train_text)


# 3. Update eval_mlp.py
with open('eval_mlp.py', 'r') as f:
    eval_text = f.read()

eval_text = eval_text.replace(
'''    st_next_pred = None
    with torch.inference_mode():
        for k in range(M_wm, M_wm + N_pred):
            if k == M_wm:
                s_in = x[:, k - 1, :state_dims]
                a_in = x[:, k - 1, state_dims+contact_dims:state_dims+contact_dims+action_dims]
            else:
                s_in = st_next_pred
                a_in = x[:, k - 1, state_dims+contact_dims:state_dims+contact_dims+action_dims]
            st_next_pred, ct_pred = world_model(s_in, a_in, x_prev=s_in)
            state_vec = st_next_pred.squeeze(0).cpu().numpy().tolist()''',
'''    st_next_pred = None
    h_t = None
    with torch.inference_mode():
        # First warmup the GRU if it exists
        if hasattr(world_model, "gru_unit"):
            for k in range(M_wm):
                s_w = x[:, k, :state_dims]
                a_w = x[:, k, state_dims+contact_dims:state_dims+contact_dims+action_dims]
                out = world_model(s_w, a_w, h_prev=h_t, x_prev=s_w)
                h_t = out[-1] # The last element is always h_next
                
        for k in range(M_wm, M_wm + N_pred):
            if k == M_wm:
                s_in = x[:, k - 1, :state_dims]
                a_in = x[:, k - 1, state_dims+contact_dims:state_dims+contact_dims+action_dims]
            else:
                s_in = st_next_pred
                a_in = x[:, k - 1, state_dims+contact_dims:state_dims+contact_dims+action_dims]
                
            if hasattr(world_model, "gru_unit"):
                out = world_model(s_in, a_in, h_prev=h_t, x_prev=s_in)
                if getattr(world_model, "with_uncertainty", False):
                    st_next_pred, ct_pred, std, h_t = out
                else:
                    st_next_pred, ct_pred, h_t = out
            else:
                if getattr(world_model, "with_uncertainty", False):
                    st_next_pred, ct_pred, std = world_model(s_in, a_in, x_prev=s_in)
                else:
                    st_next_pred, ct_pred = world_model(s_in, a_in, x_prev=s_in)
                    
            state_vec = st_next_pred.squeeze(0).cpu().numpy().tolist()'''
)

with open('eval_mlp.py', 'w') as f:
    f.write(eval_text)

# 4. Update utils.py CreateWorlModelInstance to pass with_uncertainty from config
with open('utils.py', 'r') as f:
    utils_text = f.read()

utils_text = utils_text.replace(
'''                                    lr=config['world_model_training_params']['learning_rate'],''',
'''                                    lr=config['world_model_training_params']['learning_rate'],
                                    with_uncertainty=config['world_model_arch_params'].get('with_uncertainity', False),'''
)

# And modify __init__ of RandomWorldStepGru in utils.py conceptually?
# Wait, RandomWorldStepGru doesn't accept with_uncertainty in __init__. We set it after instantiation.
utils_text = utils_text.replace(
'''                                    weight_decay=config['world_model_training_params']['weight_decay'],
                                    device=config['device'])

    return world_model''',
'''                                    weight_decay=config['world_model_training_params']['weight_decay'],
                                    device=config['device'])
    world_model.with_uncertainty = config['world_model_arch_params'].get('with_uncertainity', False)
    return world_model'''
)

with open('utils.py', 'w') as f:
    f.write(utils_text)
