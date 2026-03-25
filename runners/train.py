import json, time
from tqdm.notebook import trange, tqdm
import torch
from utils import SaveModel, CreateWorlModelInstance, get_model_name
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
        )

        # create model instance
        M, N = self.config['world_model_training_params']['M'], self.config['world_model_training_params']['N']
        state_dims = self.config['robot_params']['state_dims']
        action_dims = self.config['robot_params']['action_dims']
        world_model = CreateWorlModelInstance(self.config)
        world_model.train()
        print('World Model instantiated')
        self.get_model_params(world_model)
        training_loss = []

        # model dir 
        model_dir_name = get_model_name(model_type)
        print('this is the model name : ', model_dir_name)



        ######################### Training Starts #########################

        # Iterate over epochs
        for epoch in range(1, self.config['world_model_training_params']['epochs']+1):
            print(f'start of epoch {epoch}')
            # Iterate over batches
            for step, batch_st_ct_at in enumerate(self.data_loader):
                ht = torch.zeros((self.config['world_model_arch_params']['num_gru_layers'], 
                                  batch_st_ct_at.shape[0], 
                                  self.config['world_model_arch_params']['gru_hidden_dim'])).to(self.config['device'])
                
                x = batch_st_ct_at.to(self.config["device"]) # x -> (bs, M+N, s_dim+a_dim)
                seq_len = x.shape[1]
                batch_loss = 0
                alpha = self.config['world_model_training_params']['forecast_decay']
                # Iterate over RNN timestamps
                for t in range(seq_len-1):
                    loss_t = 0
                    if t < M-1:
                        ht = world_model.forward(torch.unsqueeze(x[:, t, :], dim=1), 
                                                 ht, predict=False) # torch.unsqueeze(x[:, t, :], dim=1) => (bs, s_dim+a_dim) -> (bs, 1, s_dim+a_dim)
                    else:
                        if t == M-1:
                            st_next_pred, ht, dist = world_model.forward(torch.unsqueeze(x[:, t, :], dim=1), ht, predict=True)
                        else:
                            st_next_pred, ht, dist = world_model.forward(torch.cat((st_next_pred, torch.unsqueeze(x[:, t, -action_dims:], dim=1)), dim=2), 
                                                                                 ht, predict=True)
                        target = torch.unsqueeze(x[:, t+1, :state_dims], dim=1)
                        loss_t = world_model.nll_loss(dist, target)
                        batch_loss += alpha * loss_t
                        alpha *= alpha
                
                batch_loss /= N

                # optimize
                world_model.optimizer.zero_grad()
                batch_loss.backward()
                world_model.optimizer.step()
                
                training_loss.append(batch_loss.item())
                # print(f'Loss at step {step} : {batch_loss.item()}')


            print(f'end of epoch {epoch}\n\n')
            
            # save model
            if epoch % self.config["save_freq"] == 0:
                model_name = f'{model_type}-epoch_{epoch}.pth'
                SaveModel(world_model, model_name, model_dir_name)
        
        ######################### Training Ends #########################


    