import json, time
from tqdm.notebook import trange, tqdm
import torch
from utils import SaveModel, CreateWorlModelInstance, get_model_name,\
                  count_parameters, SaveCkpt, LoadCkpt
import time
from data.preprocessor import load_vision_dataset

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
        if model_type == "wm_vision_rssm":
            self.update_vision(model_type)
            return
        
        # get data loader obj
        wt = self.config['world_model_training_params']
        device = self.config["device"] if torch.cuda.is_available() else "cpu"
        db_paths = self.config.get('db_paths') or [self.config['db_path']]
        self.data_loader = load_dataset(
            db_paths=db_paths,
            batch_size=wt['batch_size'],
            M=wt['M'],
            N=wt['N'],
            combined_db_path=self.config.get("combined_db_path"),
            run_mode=self.config.get("run_mode")
        )

        # create model instance
        M, N = self.config['world_model_training_params']['M'], self.config['world_model_training_params']['N']
        decay = self.config['world_model_training_params']['forecast_decay']
        state_dims = self.config['robot_params']['state_dims']
        action_dims = self.config['robot_params']['action_dims']
        world_model = CreateWorlModelInstance(self.config, model_type=model_type)
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
                                  self.config['world_model_arch_params']['gru_hidden_dim'])).to(device)
                
                x = batch_st_ct_at.to(device) # x -> (bs, M+N, s_dim+a_dim)
                seq_len = x.shape[1]
                batch_loss = 0
                alpha = 1.0
                # Iterate over RNN timestamps
                for t in range(seq_len-1):
                    loss_t = 0
                    if t < M-1:
                        ht = world_model.forward(torch.unsqueeze(x[:, t, :], dim=1), 
                                                 ht, predict=False) # torch.unsqueeze(x[:, t, :], dim=1) => (bs, s_dim+a_dim) -> (bs, 1, s_dim+a_dim)
                    else:
                        if t == M-1:
                            x_prev = torch.unsqueeze(x[:, t, :96], dim=1)
                            # std_logits, state_mean
                            st_next_pred, ht, std_logits, state_mean = world_model.forward(torch.unsqueeze(x[:, t, :], dim=1), ht, predict=True, x_prev=x_prev)
                        else:
                            st_next_pred, ht, std_logits, state_mean = world_model.forward(torch.cat((st_next_pred, torch.unsqueeze(x[:, t, -action_dims:], dim=1)), dim=2), 
                                                                                 ht, predict=True, x_prev=st_next_pred)
                        target = torch.unsqueeze(x[:, t+1, :state_dims], dim=1)
                        # loss_t = world_model.nll_loss(dist, target)
                        # print(st_next_pred.shape)
                        # print(target.shape)
                        # print("\n")
                        # loss_t = world_model.mse_loss(st_pred=torch.squeeze(st_next_pred, dim=1),
                        #                               st_true=torch.squeeze(target, dim=1))
                        loss_t = world_model.gnll_loss(state_mean=state_mean,
                                                       state_std=std_logits,
                                                       state_target=target)
                        batch_loss += alpha * loss_t
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
            if epoch % self.config["model_save_freq"] == 0:
                model_name = f'{model_type}-epoch_{epoch}.pth'
                SaveModel(world_model, model_name, model_dir_name)

            # save checkpoint
            if epoch % self.config["ckpt_save_freq"] == 0:
                model_name = f"{model_type}-ckpt-epoch_{epoch}.pth"
                SaveCkpt(world_model, model_name, model_dir_name, epoch, epoch_loss)


        
        ######################### Training Ends #########################

    def update_vision(self, model_type: str):
        vt = self.config["vision_world_model_training_params"]
        device = self.config["device"] if torch.cuda.is_available() else "cpu"
        db_paths = self.config.get("db_paths") or [self.config["db_path"]]

        self.data_loader = load_vision_dataset(
            db_paths=db_paths,
            batch_size=vt["batch_size"],
            shuffle=True,
            drop_last=True,
            traj_cache_size=vt.get("traj_cache_size", 16),
        )

        world_model = CreateWorlModelInstance(self.config, model_type=model_type)
        world_model.train()
        print("Vision World Model instantiated")
        count_parameters(world_model)

        model_dir_name = get_model_name(model_type)
        print("this is the model name : ", model_dir_name)

        epochs = vt["epochs"]
        depth_scale = vt.get("depth_scale", 50.0)
        rgb_weight = vt.get("rgb_loss_weight", 1.0)
        depth_weight = vt.get("depth_loss_weight", 1.0)
        kl_weight = vt.get("kl_weight", 1e-4)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            epoch_rgb = 0.0
            epoch_depth = 0.0
            epoch_kl = 0.0
            num_steps = 0
            print(f"start of epoch {epoch}")

            for batch in self.data_loader:
                rgb_t = batch["rgb_t"].to(device).float()
                depth_t = batch["depth_t"].to(device).float() / depth_scale
                action_t = batch["action_t"].to(device).float()
                rgb_t1 = batch["rgb_t1"].to(device).float()
                depth_t1 = batch["depth_t1"].to(device).float() / depth_scale

                outputs = world_model(
                    rgb_t=rgb_t,
                    depth_t=depth_t,
                    action_t=action_t,
                    rgb_t1=rgb_t1,
                    depth_t1=depth_t1,
                )
                losses = world_model.compute_loss(
                    outputs=outputs,
                    rgb_t1=rgb_t1,
                    depth_t1=depth_t1,
                    kl_weight=kl_weight,
                    rgb_weight=rgb_weight,
                    depth_weight=depth_weight,
                )
                total_loss = losses["total_loss"]

                world_model.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(world_model.parameters(), vt.get("grad_clip_norm", 10.0))
                world_model.optimizer.step()

                epoch_loss += total_loss.item()
                epoch_rgb += float(losses["rgb_loss"])
                epoch_depth += float(losses["depth_loss"])
                epoch_kl += float(losses["kl_loss"])
                num_steps += 1

            if num_steps > 0:
                print(
                    "epoch metrics | "
                    f"total={epoch_loss / num_steps:.6f} "
                    f"rgb={epoch_rgb / num_steps:.6f} "
                    f"depth={epoch_depth / num_steps:.6f} "
                    f"kl={epoch_kl / num_steps:.6f}"
                )
            print(f"end of epoch {epoch}\n\n")

            if epoch % self.config["save_freq"] == 0:
                model_name = f"{model_type}-epoch_{epoch}.pth"
                SaveModel(world_model, model_name, model_dir_name)


    
