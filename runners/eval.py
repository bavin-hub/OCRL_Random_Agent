import json, time
from tqdm.notebook import trange, tqdm
from utils import LoadModel, CreateWorlModelInstance, plot_graphs
import torch
import numpy as np 
import random
from itertools import chain
from data.preprocessor import load_vision_dataset



# class Inference:
#     def __init__(self, config):
#         self.config = config


#     def evaluate(self, model_type: str, load_dataset):
        
#         M = self.config['world_model_training_params']['M']
#         N = self.config['world_model_training_params']['N_pred']
#         state_dims = self.config['robot_params']['state_dims']
#         action_dims = self.config['robot_params']['action_dims']
#         contact_dims = self.config['robot_params']['contact_dims']
#         max_steps = M + N

#         # JOINT_PICK = [0, 5, 15]
#         JOINT_PICK = [i for i in range(29)]

#         # get a trajectory
#         M_tr = self.config['world_model_training_params']['M']
#         N_tr = self.config['world_model_training_params']['N']
#         db_paths = self.config.get('db_paths') or [self.config['db_path']]
#         combined_trajectory = load_dataset(
#             db_paths=db_paths,
#             combine_trajectory=True,
#             M=M_tr,
#             N=N_tr,
#         )
#         print(len(combined_trajectory[0]))
#         print('traj len : ', len(combined_trajectory))

#         start_idx = random.randint(0, len(combined_trajectory)-(M+N))
#         end_idx = start_idx + (M+N)
#         test_tuple_window = combined_trajectory[start_idx:end_idx]
#         print(len(test_tuple_window))
#         test_window = torch.tensor(np.array([list(chain.from_iterable(single_step_tuple[:1] + single_step_tuple[2:3])) 
#                                 for single_step_tuple in test_tuple_window], dtype=np.float32))
#         print('test window shape : ', test_window.shape)
#         single_trajectory = torch.unsqueeze(test_window, dim=0)
#         print(single_trajectory.shape)


        

#         ################## Evaluation starts ##################

#         # load model
#         model_name = self.config["model_name"]
#         model_dir_name = self.config["model_dir_name"]
#         world_model = CreateWorlModelInstance(self.config)
#         world_model_loaded = LoadModel(world_model, model_name, model_dir_name)

#         # generate trajectory for world model
#         world_model_loaded.eval()


        
#         ht = torch.zeros((self.config['world_model_arch_params']['num_gru_layers'], 
#                           single_trajectory.shape[0], 
#                           self.config['world_model_arch_params']['gru_hidden_dim'])).to(self.config['device'])
                
#         x = single_trajectory.to(self.config["device"])
#         single_transition: torch.Tensor

#         lin_vel_x_pred = []
#         lin_vel_y_pred = []
#         lin_vel_z_pred = []

#         ang_vel_x_pred = []
#         ang_vel_y_pred = []
#         ang_vel_z_pred = []

#         pred_joints = {j: [] for j in JOINT_PICK}

#         for t in range(max_steps-1):
#             if t < M-1:
#                 # print(.shape)
#                 single_transition = torch.squeeze(x[:, t, :], dim=0).clone().detach()
#                 # print(single_transition.shape)
#                 single_transition = single_transition[:96].to('cpu').numpy().tolist()
#                 ht = world_model.forward(torch.unsqueeze(x[:, t, :], dim=1), ht, predict=False)           
#             else:
#                 if t == M-1:
#                     # print('there')
#                     st_next_pred, ht = world_model.forward(torch.unsqueeze(x[:, t, :], dim=1), ht, predict=True, sample=True)
#                 else:
#                     # print('after')
#                     st_next_pred, ht = world_model.forward(torch.cat((st_next_pred, torch.unsqueeze(x[:, t, -action_dims:], dim=1)), dim=2), 
#                                                                       ht, predict=True, sample=True)
    
#                 single_transition = torch.squeeze(st_next_pred[-1], dim=0).clone().detach()
#                 single_transition = single_transition.to('cpu').numpy().tolist()
          
#             state_vec = single_transition
#             pred_velocities = state_vec[:6]
#             lin_vel_x_pred.append(pred_velocities[0])
#             lin_vel_y_pred.append(pred_velocities[1])
#             lin_vel_z_pred.append(pred_velocities[2])
#             ang_vel_x_pred.append(pred_velocities[3])
#             ang_vel_y_pred.append(pred_velocities[4])
#             ang_vel_z_pred.append(pred_velocities[5])
#             for j in JOINT_PICK:
#                 pred_joints[j].append(state_vec[9 + j])

#         ################## Evaluation ends ##################

#         # print('pred vel x len : ',len(lin_vel_x_pred))

#         # plot trajectory graph
#         lin_vel_x = []
#         lin_vel_y = []
#         lin_vel_z = []

#         ang_vel_x = []
#         ang_vel_y = []
#         ang_vel_z = []

#         true_joints = {j: [] for j in JOINT_PICK}

#         true_traj = np.squeeze(single_trajectory.numpy())
#         print(true_traj.shape)
#         print('here')

#         for i in range(max_steps-1):
#             transition = true_traj[i, :]
#             state = transition[:96].tolist()
#             velocities = state[:6]
#             lin_vel_x.append(velocities[0])
#             lin_vel_y.append(velocities[1])
#             lin_vel_z.append(velocities[2])
#             ang_vel_x.append(velocities[3])
#             ang_vel_y.append(velocities[4])
#             ang_vel_z.append(velocities[5])
#             for j in JOINT_PICK:
#                 true_joints[j].append(state[9 + j])

#         lin_vel_true = [lin_vel_x,
#                         lin_vel_y,
#                         lin_vel_z]
        
#         lin_vel_preds = [lin_vel_x_pred,
#                          lin_vel_y_pred,
#                          lin_vel_z_pred]
        
#         ang_vel_true = [ang_vel_x,
#                         ang_vel_y,
#                         ang_vel_z]
        
#         ang_vel_preds = [ang_vel_x_pred,
#                          ang_vel_y_pred,
#                          ang_vel_z_pred]

#         plot_graphs(lin_vel_true=lin_vel_true,
#                     lin_vel_preds=lin_vel_preds,
#                     ang_vel_true=ang_vel_true,
#                     ang_vel_preds=ang_vel_preds,
#                     true_joints=true_joints,
#                     pred_joints=pred_joints,
#                     M=M,
#                     model_dir_name=model_dir_name,
#                     model_name=model_name)




class Inference:
    def __init__(self, config):
        self.config = config


    def evaluate(self, model_type: str, load_dataset):
        if model_type in ("wm_vision_rssm", "wm_vision"):
            self.evaluate_vision(model_type)
            return
        
        M = self.config['world_model_training_params']['M']
        N = self.config['world_model_training_params']['N_pred']
        state_dims = self.config['robot_params']['state_dims']
        action_dims = self.config['robot_params']['action_dims']
        contact_dims = self.config['robot_params']['contact_dims']
        max_steps = M + N

        # JOINT_PICK = [0, 5, 15]
        JOINT_PICK = [i for i in range(29)]

        # get a trajectory
        M_tr = self.config['world_model_training_params']['M']
        N_tr = self.config['world_model_training_params']['N']
        db_paths = self.config.get('db_paths') or [self.config['db_path']]
        combined_trajectory = load_dataset(
            db_paths=db_paths,
            combine_trajectory=True,
            M=M_tr,
            N=N_tr,
        )
        print(len(combined_trajectory[0]))
        print('traj len : ', len(combined_trajectory))

        start_idx = random.randint(0, len(combined_trajectory)-(M+N))
        end_idx = start_idx + (M+N)
        test_tuple_window = combined_trajectory[start_idx:end_idx]
        print(len(test_tuple_window))
        test_window = torch.tensor(np.array([list(chain.from_iterable(single_step_tuple[:1] + single_step_tuple[2:3])) 
                                for single_step_tuple in test_tuple_window], dtype=np.float32))
        print('test window shape : ', test_window.shape)
        single_trajectory = torch.unsqueeze(test_window, dim=0)
        print(single_trajectory.shape)


        

        ################## Evaluation starts ##################

        # load model
        model_name = self.config["model_name"]
        model_dir_name = self.config["model_dir_name"]
        world_model = CreateWorlModelInstance(self.config, model_type=model_type)
        world_model_loaded = LoadModel(world_model, model_name, model_dir_name)

        # generate trajectory for world model
        world_model_loaded.eval()


        
        ht = torch.zeros((self.config['world_model_arch_params']['num_gru_layers'], 
                          single_trajectory.shape[0], 
                          self.config['world_model_arch_params']['gru_hidden_dim'])).to(self.config['device'])
                
        x = single_trajectory.to(self.config["device"])
        single_transition: torch.Tensor

        lin_vel_x_pred = []
        lin_vel_y_pred = []
        lin_vel_z_pred = []

        ang_vel_x_pred = []
        ang_vel_y_pred = []
        ang_vel_z_pred = []

        pred_joints = {j: [] for j in JOINT_PICK}

        for t in range(max_steps-1):
            if t < M-1:
                # print(.shape)
                single_transition = torch.squeeze(x[:, t, :], dim=0).clone().detach()
                # print(single_transition.shape)
                single_transition = single_transition[:96].to('cpu').numpy().tolist()
                ht = world_model.forward(torch.unsqueeze(x[:, t, :], dim=1), ht, predict=False)           
            else:
                if t == M-1:
                    # print('there')
                    x_prev = torch.unsqueeze(x[:, t, :96], dim=1)
                    st_next_pred, ht = world_model.forward(torch.unsqueeze(x[:, t, :], dim=1), ht, predict=True, sample=True, x_prev=x_prev)
                else:
                    # print('after')
                    st_next_pred, ht = world_model.forward(torch.cat((st_next_pred, torch.unsqueeze(x[:, t, -action_dims:], dim=1)), dim=2), 
                                                                      ht, predict=True, sample=True, x_prev=st_next_pred)
    
                single_transition = torch.squeeze(st_next_pred[-1], dim=0).clone().detach()
                single_transition = single_transition.to('cpu').numpy().tolist()
          
            state_vec = single_transition
            pred_velocities = state_vec[:6]
            lin_vel_x_pred.append(pred_velocities[0])
            lin_vel_y_pred.append(pred_velocities[1])
            lin_vel_z_pred.append(pred_velocities[2])
            ang_vel_x_pred.append(pred_velocities[3])
            ang_vel_y_pred.append(pred_velocities[4])
            ang_vel_z_pred.append(pred_velocities[5])
            for j in JOINT_PICK:
                pred_joints[j].append(state_vec[9 + j])

        ################## Evaluation ends ##################

        # print('pred vel x len : ',len(lin_vel_x_pred))

        # plot trajectory graph
        lin_vel_x = []
        lin_vel_y = []
        lin_vel_z = []

        ang_vel_x = []
        ang_vel_y = []
        ang_vel_z = []

        true_joints = {j: [] for j in JOINT_PICK}

        true_traj = np.squeeze(single_trajectory.numpy())
        print(true_traj.shape)
        print('here')

        for i in range(max_steps-1):
            transition = true_traj[i, :]
            state = transition[:96].tolist()
            velocities = state[:6]
            lin_vel_x.append(velocities[0])
            lin_vel_y.append(velocities[1])
            lin_vel_z.append(velocities[2])
            ang_vel_x.append(velocities[3])
            ang_vel_y.append(velocities[4])
            ang_vel_z.append(velocities[5])
            for j in JOINT_PICK:
                true_joints[j].append(state[9 + j])

        lin_vel_true = [lin_vel_x,
                        lin_vel_y,
                        lin_vel_z]
        
        lin_vel_preds = [lin_vel_x_pred,
                         lin_vel_y_pred,
                         lin_vel_z_pred]
        
        ang_vel_true = [ang_vel_x,
                        ang_vel_y,
                        ang_vel_z]
        
        ang_vel_preds = [ang_vel_x_pred,
                         ang_vel_y_pred,
                         ang_vel_z_pred]

        plot_graphs(lin_vel_true=lin_vel_true,
                    lin_vel_preds=lin_vel_preds,
                    ang_vel_true=ang_vel_true,
                    ang_vel_preds=ang_vel_preds,
                    true_joints=true_joints,
                    pred_joints=pred_joints,
                    M=M,
                    model_dir_name=model_dir_name,
                    model_name=model_name)

    def evaluate_vision(self, model_type: str):
        vt = self.config["vision_world_model_training_params"]
        device = self.config["device"] if torch.cuda.is_available() else "cpu"
        db_paths = self.config.get("db_paths") or [self.config["db_path"]]

        data_loader = load_vision_dataset(
            db_paths=db_paths,
            batch_size=vt.get("eval_batch_size", 32),
            shuffle=False,
            drop_last=False,
            traj_cache_size=vt.get("traj_cache_size", 16),
            seq_len=vt.get("seq_len", 8),
        )

        model_name = self.config["model_name"]
        model_dir_name = self.config["model_dir_name"]
        world_model = CreateWorlModelInstance(self.config, model_type=model_type)
        world_model = LoadModel(world_model, model_name, model_dir_name)
        world_model.eval()

        depth_scale  = vt.get("depth_scale", 50.0)
        free_bits    = vt.get("free_bits", 1.0)
        kl_balance   = vt.get("kl_balance", 0.8)
        rgb_weight   = vt.get("rgb_loss_weight", 1.0)
        depth_weight = vt.get("depth_loss_weight", 1.0)
        lpips_weight = vt.get("lpips_weight", 0.5)
        uses_posterior = model_type != "wm_vision"

        total_loss = rgb_loss = depth_loss = kl_loss = perceptual_loss = 0.0
        with torch.no_grad():
            for batch in data_loader:
                rgb_t    = batch["rgb_t"].to(device).float()
                depth_t  = batch["depth_t"].to(device).float() / depth_scale
                action_t = batch["action_t"].to(device).float()
                prop_t   = batch["prop_t"].to(device).float()
                rgb_t1   = batch["rgb_t1"].to(device).float()
                depth_t1 = batch["depth_t1"].to(device).float() / depth_scale

                seq_len = rgb_t.shape[1]
                step_loss = step_rgb = step_depth = step_kl = step_lpips = 0.0

                if not uses_posterior:
                    # wm_vision: full-sequence forward
                    outputs = world_model(
                        rgb_t=rgb_t, depth_t=depth_t,
                        action_t=action_t, prop_t=prop_t,
                    )
                    losses = world_model.compute_loss(
                        outputs=outputs, rgb_t1=rgb_t1, depth_t1=depth_t1,
                        rgb_weight=rgb_weight, depth_weight=depth_weight,
                    )
                    step_loss  = float(losses["total_loss"])
                    step_rgb   = float(losses["rgb_loss"])
                    step_depth = float(losses["depth_loss"])
                    step_kl    = float(losses["kl_loss"])
                    step_lpips = float(losses.get("perceptual_loss", 0.0))
                else:
                    # RSSM: step-by-step
                    hidden_state = None
                    for t in range(seq_len):
                        outputs = world_model(
                            rgb_t=rgb_t[:, t], depth_t=depth_t[:, t],
                            action_t=action_t[:, t],
                            rgb_t1=rgb_t1[:, t], depth_t1=depth_t1[:, t],
                            hidden_state=hidden_state, prop_t=prop_t[:, t],
                        )
                        losses = world_model.compute_loss(
                            outputs=outputs, rgb_t1=rgb_t1[:, t],
                            depth_t1=depth_t1[:, t],
                            rgb_weight=rgb_weight, depth_weight=depth_weight,
                            free_bits=free_bits, kl_balance=kl_balance,
                            lpips_weight=lpips_weight,
                        )
                        hidden_state = outputs["hidden_next"]
                        step_loss  += float(losses["total_loss"])
                        step_rgb   += float(losses["rgb_loss"])
                        step_depth += float(losses["depth_loss"])
                        step_kl    += float(losses["kl_loss"])
                        step_lpips += float(losses.get("perceptual_loss", 0.0))
                    step_loss  /= seq_len
                    step_rgb   /= seq_len
                    step_depth /= seq_len
                    step_kl    /= seq_len
                    step_lpips /= seq_len

                total_loss      += step_loss
                rgb_loss        += step_rgb
                depth_loss      += step_depth
                kl_loss         += step_kl
                perceptual_loss += step_lpips

        n = len(data_loader)
        print("Vision eval metrics (avg over dataset):")
        print(f"  total_loss={total_loss / n:.6f}")
        print(f"  rgb_loss={rgb_loss / n:.6f}")
        print(f"  depth_loss={depth_loss / n:.6f}")
        print(f"  lpips_loss={perceptual_loss / n:.6f}")
        print(f"  kl_loss={kl_loss / n:.6f}")

        




            
            



            
