from models.world_model import RandomWorldStepGru
import json, time
from tqdm.notebook import trange, tqdm
from utils import LoadModel, CreateWorlModelInstance
import matplotlib.pyplot as plt
import torch
import numpy as np 
import random
from itertools import chain

class Inference:
    def __init__(self, config):
        self.config = config




    def evaluate(self, model_type: str, load_dataset):
        
        M = self.config['world_model_training_params']['M']
        N = 20
        state_dims = self.config['robot_params']['state_dims']
        action_dims = self.config['robot_params']['action_dims']
        contact_dims = self.config['robot_params']['contact_dims']
        max_steps = M + N

        # get a trajectory
        combined_trajectory = load_dataset(db_path=self.config["db_path"], combine_trajectory=True)
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


        # load model
        model_path = "/home/bavin/cmu/sem2/ocrl/OCRL_Random_Agent/logs/models/world_model-epoch_2500.pth"
        world_model = CreateWorlModelInstance(self.config)
        world_model_loaded = LoadModel(world_model, model_path)

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
        
        
        for t in range(max_steps-1):
            if t < M-1:
                # print(.shape)
                single_transition = torch.squeeze(x[:, t, :], dim=0).clone().detach()
                print(single_transition.shape)
                single_transition = single_transition[:96].to('cpu').numpy().tolist()
                ht = world_model.forward(torch.unsqueeze(x[:, t, :], dim=1), ht, predict=False)           
            else:
                if t == M-1:
                    # print('there')
                    st_next_pred, ht = world_model.forward(torch.unsqueeze(x[:, t, :], dim=1), ht, predict=True, sample=True)
                else:
                    # print('after')
                    st_next_pred, ht = world_model.forward(torch.cat((st_next_pred, torch.unsqueeze(x[:, t, -action_dims:], dim=1)), dim=2), 
                                                                      ht, predict=True, sample=True)
    
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

        print('pred vel x len : ',len(lin_vel_x_pred))

        # plot trajectory graph
        lin_vel_x = []
        lin_vel_y = []
        lin_vel_z = []

        ang_vel_x = []
        ang_vel_y = []
        ang_vel_z = []

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


        x = [_ + 1 for _ in range(len(lin_vel_x))] 
        y = lin_vel_x
        y_pred = lin_vel_x_pred

        plt.plot(x, y, label='Solid Line')
        plt.plot(x, y_pred, label='Dotted Line', linestyle=':')
        plt.show()











    # def evaluate(self, model_type: str, load_dataset):
        
    #     M = self.config['world_model_training_params']['M']
    #     # , self.config['world_model_training_params']['N']
    #     N = 20
    #     state_dims = self.config['robot_params']['state_dims']
    #     action_dims = self.config['robot_params']['action_dims']
    #     contact_dims = self.config['robot_params']['contact_dims']
    #     max_steps = M + N

    #     # get a trajectory
    #     combined_trajectory = load_dataset(db_path=self.config["db_path"], combine_trajectory=True)
    #     print(len(combined_trajectory[0]))
    #     print('traj len : ', len(combined_trajectory))
    #     # print(single_batch.shape)
    #     # trajectory_batch = single_batch.reshape(single_batch.shape[0], -1)

    #     start_idx = random.randint(0, len(combined_trajectory)-(M+N))
    #     end_idx = start_idx + (M+N)
    #     test_tuple_window = combined_trajectory[start_idx:end_idx]
    #     print(len(test_tuple_window))
    #     test_window = torch.tensor(np.array([list(chain.from_iterable(single_step_tuple[:1] + single_step_tuple[2:3])) 
    #                             for single_step_tuple in test_tuple_window], dtype=np.float32))
    #     print('test window shape : ', test_window.shape)
    #     single_trajectory = torch.unsqueeze(test_window, dim=0)
    #     print(single_trajectory.shape)

    #     # get a single batch 
    #     # single_batch = load_dataset(db_path=self.config["db_path"], single_batch=True)
    #     # single_trajectory = torch.unsqueeze(single_batch[2], dim=0)
    #     # print('single batch shape : ', single_batch.shape)
    #     # print('single trajectory shape : ', single_trajectory.shape)
    #     # print(torch.max(single_batch))
    #     # [:, :, 96:]

    #     # load model
    #     model_path = "/home/bavin/cmu/sem2/ocrl/OCRL_Random_Agent/logs/models/world_model-epoch_2500.pth"
    #     world_model = CreateWorlModelInstance(self.config)
    #     world_model_loaded = LoadModel(world_model, model_path)

    #     # generate trajectory for world model
    #     world_model_loaded.eval()


        
    #     ht = torch.zeros((self.config['world_model_arch_params']['num_gru_layers'], 
    #                       single_trajectory.shape[0], 
    #                       self.config['world_model_arch_params']['gru_hidden_dim'])).to(self.config['device'])
                
    #     x = single_trajectory.to(self.config["device"])
    #     single_transition: torch.Tensor

    #     lin_vel_x_pred = []
    #     lin_vel_y_pred = []
    #     lin_vel_z_pred = []

    #     ang_vel_x_pred = []
    #     ang_vel_y_pred = []
    #     ang_vel_z_pred = []
        
        
    #     for t in range(max_steps-1):
    #         if t < M-1:
    #             # print(.shape)
    #             single_transition = torch.squeeze(x[:, t, :], dim=0).clone().detach()
    #             print(single_transition.shape)
    #             single_transition = single_transition[:96].to('cpu').numpy().tolist()
    #             ht = world_model.forward(torch.unsqueeze(x[:, t, :], dim=1), ht, predict=False)           
    #         else:
    #             if t == M-1:
    #                 # print('there')
    #                 st_next_pred, ht = world_model.forward(torch.unsqueeze(x[:, t, :], dim=1), ht, predict=True, sample=True)
    #             else:
    #                 # print('after')
    #                 st_next_pred, ht = world_model.forward(torch.cat((st_next_pred, torch.unsqueeze(x[:, t, -action_dims:], dim=1)), dim=2), 
    #                                                                   ht, predict=True, sample=True)
    
    #             single_transition = torch.squeeze(st_next_pred[-1], dim=0).clone().detach()
    #             single_transition = single_transition.to('cpu').numpy().tolist()
    #             # batch_loss += (alpha * torch.sum(world_model.state_discrepancy(st_next_pred, torch.unsqueeze(x[:, t+1, :state_dims], dim=1)) + 
    #             #                                     world_model.contact_discrepancy(ct_next_pred, torch.unsqueeze(x[:, t+1, state_dims:state_dims+contact_dims], dim=1))))
    #             # alpha *= alpha

    #         # print(single_transition.shape)
    #         state_vec = single_transition
    #         pred_velocities = state_vec[:6]
    #         lin_vel_x_pred.append(pred_velocities[0])
    #         lin_vel_y_pred.append(pred_velocities[1])
    #         lin_vel_z_pred.append(pred_velocities[2])
    #         ang_vel_x_pred.append(pred_velocities[3])
    #         ang_vel_y_pred.append(pred_velocities[4])
    #         ang_vel_z_pred.append(pred_velocities[5])

    #     print('pred vel x len : ',len(lin_vel_x_pred))

    #     # plot trajectory graph
    #     lin_vel_x = []
    #     lin_vel_y = []
    #     lin_vel_z = []

    #     ang_vel_x = []
    #     ang_vel_y = []
    #     ang_vel_z = []

    #     true_traj = np.squeeze(single_trajectory.numpy())
    #     print(true_traj.shape)
    #     print('here')

    #     for i in range(max_steps-1):
    #         transition = true_traj[i, :]
    #         state = transition[:96].tolist()
    #         velocities = state[:6]
    #         lin_vel_x.append(velocities[0])
    #         lin_vel_y.append(velocities[1])
    #         lin_vel_z.append(velocities[2])
    #         ang_vel_x.append(velocities[3])
    #         ang_vel_y.append(velocities[4])
    #         ang_vel_z.append(velocities[5])

    #     # for transition in combined_trajectory[80:120]:
    #     #     state, contact, target_action, policy_action = transition
            
    #     #     velocities = state[:6]
    #     #     lin_vel_x.append(velocities[0])
    #     #     lin_vel_y.append(velocities[1])
    #     #     lin_vel_z.append(velocities[2])
    #     #     ang_vel_x.append(velocities[3])
    #     #     ang_vel_y.append(velocities[4])
    #     #     ang_vel_z.append(velocities[5])


    #     print(len(lin_vel_x))
    #     print(len(lin_vel_x_pred))

    #     x = [_ + 1 for _ in range(len(lin_vel_x))] 
    #     y = lin_vel_x
    #     y_pred = lin_vel_x_pred

    #     plt.plot(x, y, label='Solid Line')
    #     plt.plot(x, y_pred, label='Dotted Line', linestyle=':')
    #     plt.show()



    # def evaluate(self, model_type: str, load_dataset):
        
    #     M, N = self.config['world_model_training_params']['M'], self.config['world_model_training_params']['N']
    #     state_dims = self.config['robot_params']['state_dims']
    #     action_dims = self.config['robot_params']['action_dims']
    #     contact_dims = self.config['robot_params']['contact_dims']
    #     max_steps = M + N

    #     # get a trajectory
    #     combined_trajectory = load_dataset(db_path=self.config["db_path"], combine_trajectory=True)
    #     print(len(combined_trajectory[0]))
    #     print('traj len : ', len(combined_trajectory))
    #     # print(single_batch.shape)
    #     # trajectory_batch = single_batch.reshape(single_batch.shape[0], -1)

    #     # get a single batch 
    #     single_batch = load_dataset(db_path=self.config["db_path"], single_batch=True)
    #     single_trajectory = torch.unsqueeze(single_batch[0], dim=0)
    #     print('single batch shape : ', single_batch.shape)
    #     print('single trajectory shape : ', single_trajectory.shape)

    #     # load model
    #     model_path = "/home/bavin/cmu/sem2/ocrl/OCRL_Random_Agent/logs/models/world_model-epoch_2500.pth"
    #     world_model = CreateWorlModelInstance(self.config)
    #     world_model_loaded = LoadModel(world_model, model_path)

    #     # generate trajectory for world model
    #     world_model_loaded.eval()


        
    #     ht = torch.zeros((self.config['world_model_arch_params']['num_gru_layers'], 
    #                       single_trajectory.shape[0], 
    #                       self.config['world_model_arch_params']['gru_hidden_dim'])).to(self.config['device'])
                
    #     x = single_trajectory.to(self.config["device"])
    #     single_transition: torch.Tensor

    #     lin_vel_x_pred = []
    #     lin_vel_y_pred = []
    #     lin_vel_z_pred = []

    #     ang_vel_x_pred = []
    #     ang_vel_y_pred = []
    #     ang_vel_z_pred = []
        
        
    #     for t in range(max_steps-1):
    #         if t < M-1:
    #             # print(.shape)
    #             single_transition = torch.squeeze(x[:, t, :], dim=0)
    #             single_transition = single_transition[:96].to('cpu').numpy().tolist()
    #             ht = world_model.forward(torch.unsqueeze(x[:, t, :], dim=1), ht, predict=False)           
    #         else:
    #             if t == M-1:
    #                 # print('there')
    #                 st_next_pred, ht, ct_next_pred = world_model.forward(torch.unsqueeze(x[:, t, :], dim=1), ht, predict=True, sample=True)
    #             else:
    #                 # print('after')
    #                 st_next_pred, ht, ct_next_pred = world_model.forward(torch.cat((st_next_pred, ct_next_pred, torch.unsqueeze(x[:, t, -action_dims:], dim=1)), dim=2), 
    #                                                                         ht, predict=True, sample=True)
    
    #             single_transition = torch.squeeze(st_next_pred[-1].clone().detach(), dim=0)
    #             single_transition = single_transition.to('cpu').numpy().tolist()
    #             # batch_loss += (alpha * torch.sum(world_model.state_discrepancy(st_next_pred, torch.unsqueeze(x[:, t+1, :state_dims], dim=1)) + 
    #             #                                     world_model.contact_discrepancy(ct_next_pred, torch.unsqueeze(x[:, t+1, state_dims:state_dims+contact_dims], dim=1))))
    #             # alpha *= alpha

    #         # print(single_transition.shape)
    #         state_vec = single_transition
    #         pred_velocities = state_vec[:6]
    #         lin_vel_x_pred.append(pred_velocities[0])
    #         lin_vel_y_pred.append(pred_velocities[1])
    #         lin_vel_z_pred.append(pred_velocities[2])
    #         ang_vel_x_pred.append(pred_velocities[3])
    #         ang_vel_y_pred.append(pred_velocities[4])
    #         ang_vel_z_pred.append(pred_velocities[5])

    #     print('pred vel x len : ',len(lin_vel_x_pred))

    #     # plot trajectory graph
    #     lin_vel_x = []
    #     lin_vel_y = []
    #     lin_vel_z = []

    #     ang_vel_x = []
    #     ang_vel_y = []
    #     ang_vel_z = []

    #     for transition in combined_trajectory[:max_steps]:
    #         state, contact, target_action, policy_action = transition
            
    #         velocities = state[:6]
    #         lin_vel_x.append(velocities[0])
    #         lin_vel_y.append(velocities[1])
    #         lin_vel_z.append(velocities[2])
    #         ang_vel_x.append(velocities[3])
    #         ang_vel_y.append(velocities[4])
    #         ang_vel_z.append(velocities[5])


    #     print(type(lin_vel_x[0]))
    #     print(min(lin_vel_x))

    #     x = [_ + 1 for _ in range(len(lin_vel_x)-1)] 
    #     y = lin_vel_x[:39]
    #     y_pred = lin_vel_x_pred

    #     plt.plot(x, y, label='Solid Line')
    #     plt.plot(x, y_pred, label='Dotted Line', linestyle=':')
    #     plt.show()


        
        

            
            



            