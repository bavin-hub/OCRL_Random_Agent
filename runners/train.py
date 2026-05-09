import json, time
from tqdm.notebook import trange, tqdm
import torch
from utils import SaveModel, CreateGruWMInstance, get_model_name,\
                  count_parameters, SaveCkpt, LoadCkpt
import time

# only state-action pair
class Trainer:

    def __init__(self, config: dict):
        self.config = config
        # self.data_loader = load_dataset(db_path=self.config["db_path"])


    def get_model_params(self, model):
        num_model_params = 0
        for param in model.parameters():
            num_model_params += param.flatten().shape[0]
        
        print('Total params in the world model : ', num_model_params)


    def update(self, model_type, load_dataset):
        
        # get data loader obj
        wt = self.config['world_model_training_params']
        db_paths = self.config.get('db_paths') or [self.config['db_path']]
        self.data_loader = load_dataset(
            db_paths=db_paths,
            batch_size=wt['batch_size'],
            M=wt['M'],
            N=wt['N'],
            combined_db_path=self.config.get("combined_db_path"),
            run_mode=self.config.get("run_mode"),
            mean=self.config.get("mean_state_action"),
            std=self.config.get("std_state_action")
        )

        # create model instance
        M, N = self.config['world_model_training_params']['M'], self.config['world_model_training_params']['N']
        decay = self.config['world_model_training_params']['forecast_decay']
        state_dims = self.config['robot_params']['state_dims']
        action_dims = self.config['robot_params']['action_dims']
        world_model = CreateGruWMInstance(self.config)
        world_model.train()
        print('World Model instantiated')
        # self.get_model_params(world_model)
        count_parameters(world_model)
        training_loss = []

        # model dir 
        model_dir_name = get_model_name(model_type)
        print('this is the model name : ', model_dir_name)


        # load checkpoints
        if self.config["use_ckpt"]:
            world_model = LoadCkpt(world_model,
                                   self.config["ckpt_name"],
                                   self.config["ckpt_dir"]) 



        ######################### Training Starts #########################

        # Iterate over epochs
        for epoch in range(1, self.config['world_model_training_params']['epochs']+1):
            print(f'start of epoch {epoch}')
            # Iterate over batches
            epoch_loss = 0.0
            for step, batch_st_ct_at in enumerate(self.data_loader):
                ht = torch.zeros((self.config['world_model_arch_params']['num_gru_layers'], 
                                  batch_st_ct_at.shape[0], 
                                  self.config['world_model_arch_params']['gru_hidden_dim'])).to(self.config['device'])
                
                x = batch_st_ct_at.to(self.config["device"]) # x -> (bs, M+N, s_dim+a_dim)
                seq_len = x.shape[1]
                batch_loss = 0
                alpha = 1.0
                # Iterate over RNN timestamps
                for t in range(seq_len-1):
                    loss_t = 0
                    
                    # Extract current state and action (ignore contacts for input)
                    # x is structured as [states (96), contacts (30), actions (12)]
                    st_t = torch.unsqueeze(x[:, t, :state_dims], dim=1)
                    at_t = torch.unsqueeze(x[:, t, state_dims+30:state_dims+30+action_dims], dim=1)
                    input_t = torch.cat((st_t, at_t), dim=-1)
                    
                    if t < M-1:
                        ht = world_model.forward(input_t, ht, predict=False) 
                    else:
                        if t == M-1:
                            x_prev = st_t
                            st_next_pred, ht, std_logits, mu_t_next, contact_pred = world_model.forward(input_t, ht, predict=True, x_prev=x_prev)
                        else:
                            # Feedback the predicted state along with the real action
                            input_t_pred = torch.cat((st_next_pred, at_t), dim=-1)
                            st_next_pred, ht, std_logits, mu_t_next, contact_pred = world_model.forward(input_t_pred, ht, predict=True, x_prev=st_next_pred)
                            
                        target = torch.unsqueeze(x[:, t+1, :state_dims], dim=1)
                        c_target = torch.unsqueeze(x[:, t+1, state_dims:state_dims+30], dim=1)
                        
                        # GNLL loss for state prediction (with uncertainty)
                        loss_t = world_model.gnll_loss(state_mean=torch.squeeze(mu_t_next, dim=1),
                                                       state_std=torch.squeeze(std_logits, dim=1),
                                                       state_target=torch.squeeze(target, dim=1))
                        
                        # MSE loss for contact prediction
                        loss_c = world_model.mse_loss(st_pred=torch.squeeze(contact_pred, dim=1),
                                                      st_true=torch.squeeze(c_target, dim=1))
                                                      
                        contact_weight = self.config['world_model_training_params'].get('contact_loss_weight', 0.5)
                        batch_loss += alpha * (loss_t + contact_weight * loss_c)
                        alpha *= decay
                
                batch_loss /= N

                # optimize
                world_model.optimizer.zero_grad()
                batch_loss.backward()
                world_model.optimizer.step()

                epoch_loss += batch_loss
                training_loss.append(batch_loss.item())
                # print(f'Loss at step {step} : {batch_loss.item()}')


            print(f'end of epoch {epoch}\n\n')
            
            # save model
            if epoch % self.config["model_save_freq"] == 0 or epoch == 1:
                model_name = f'{model_type}-epoch_{epoch}.pth'
                SaveModel(world_model, model_name, model_dir_name)

            # save checkpoint
            if epoch % self.config["ckpt_save_freq"] == 0 or epoch == 1:
                model_name = f"{model_type}-ckpt-epoch_{epoch}.pth"
                SaveCkpt(world_model, model_name, model_dir_name, epoch, epoch_loss)


        
        ######################### Training Ends #########################


    