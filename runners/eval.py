from models.world_model import RandomWorldStepGru
import json, time
from tqdm.notebook import trange, tqdm
from utils import LoadModel, CreateWorlModelInstance, plot_graphs
import matplotlib.pyplot as plt
import torch
import numpy as np 
import random
from itertools import chain



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
        world_model = CreateWorlModelInstance(self.config)
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

        




            
            



            