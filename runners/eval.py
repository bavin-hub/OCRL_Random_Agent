from models.world_model import RandomWorldStepGru
import json, time
from tqdm.notebook import trange, tqdm
from utils import LoadModel, CreateGruWMInstance, plot_graphs, z_norm
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
#         world_model = CreateGruWMInstance(self.config)
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
        mean = np.array([self.config.get("mean_state_action")])
        std = np.array([self.config.get("std_state_action")])

        # JOINT_PICK = [0, 5, 15]
        JOINT_PICK = [i for i in range(29)]

        # Read directly from combined_transitions.db (same ordering as eval_mlp.py)
        import sqlite3 as _sql
        combined_db_path = self.config.get('combined_db_path')
        conn = _sql.connect(combined_db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM PretrainingData")
            rows = cur.fetchall()
        finally:
            conn.close()
        all_steps = []
        for row in rows:
            blob = json.loads(row[0])
            traj_key = next(iter(blob.keys()))
            all_steps.extend(blob[traj_key])

        eval_seed = self.config.get('eval_seed', 42)
        rng = random.Random(eval_seed)
        start_idx = rng.randint(0, len(all_steps) - (M + N))
        print(f"[eval_gru] seed={eval_seed}  start_idx={start_idx}  total_steps={len(all_steps)}")
        end_idx = start_idx + (M + N)
        test_tuple_window = all_steps[start_idx:end_idx]
        print(len(test_tuple_window))
        test_window = np.array([list(chain.from_iterable(single_step_tuple[:2] + single_step_tuple[2:3])) 
                                for single_step_tuple in test_tuple_window], dtype=np.float32)
        
        # Apply z_norm selectively to ignore contacts
        state_action_window = np.concatenate([test_window[:, :96], test_window[:, 126:]], axis=1)
        normed_state_action = z_norm(state_action_pair=state_action_window, mean=mean, std=std)
        test_window[:, :96] = normed_state_action[:, :96]
        test_window[:, 126:] = normed_state_action[:, 96:]
        
        test_window = torch.tensor(test_window)
        print('test window shape : ', test_window.shape)
        single_trajectory = torch.unsqueeze(test_window, dim=0)
        print(single_trajectory.shape)


        

        ################## Evaluation starts ##################

        # load model
        model_name = self.config["model_name"]
        model_dir_name = self.config["model_dir_name"]
        world_model = CreateGruWMInstance(self.config)

        wm_ckpt = self.config.get("wm_checkpoint")
        if wm_ckpt:
            # Direct path loading (supports both raw state_dict and checkpoint dict)
            payload = torch.load(wm_ckpt, weights_only=True, map_location=self.config['device'])
            if isinstance(payload, dict) and "model_state_dict" in payload:
                world_model.load_state_dict(payload["model_state_dict"])
            else:
                world_model.load_state_dict(payload)
            print(f"\n#########\nModel loaded from {wm_ckpt}\n#########\n")
            world_model_loaded = world_model
        else:
            world_model_loaded = LoadModel(world_model, model_name, model_dir_name)

        # generate trajectory for world model
        world_model_loaded.eval()


        
        ht = torch.zeros((self.config['world_model_arch_params']['num_gru_layers'], 
                          single_trajectory.shape[0], 
                          self.config['world_model_arch_params']['gru_hidden_dim'])).to(self.config['device'])
                
        x = single_trajectory.to(self.config["device"])
        single_transition: torch.Tensor

        # Load standard deviation and mean properly for un-normalization
        mean = self.config.get('mean_state_action')
        std = self.config.get('std_state_action')
        mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, -1)
        std_arr = np.asarray(std, dtype=np.float32).reshape(1, -1)
        if mean_arr.shape[1] == 125:
            C_mean = np.zeros((1, 30), dtype=np.float32)
            C_std = np.ones((1, 30), dtype=np.float32)
            mean_arr = np.concatenate([mean_arr[:, :96], C_mean, mean_arr[:, 96:]], axis=1)
            std_arr = np.concatenate([std_arr[:, :96], C_std, std_arr[:, 96:]], axis=1)

        state_mean = mean_arr[0, :state_dims]
        state_std = std_arr[0, :state_dims]
        contact_mean = mean_arr[0, state_dims:state_dims+contact_dims]
        contact_std = std_arr[0, state_dims:state_dims+contact_dims]

        lin_vel_x_pred, lin_vel_y_pred, lin_vel_z_pred = [], [], []
        ang_vel_x_pred, ang_vel_y_pred, ang_vel_z_pred = [], [], []
        grav_x_pred, grav_y_pred, grav_z_pred = [], [], []
        pred_joints = {j: [] for j in JOINT_PICK}
        pred_joint_vel = {j: [] for j in JOINT_PICK}
        pred_joint_tau = {j: [] for j in JOINT_PICK}
        pred_contacts = {j: [] for j in range(contact_dims)}

        n_joints = 29
        off_qd = 9 + n_joints
        off_tau = 9 + 2 * n_joints

        def _append_state(state_vec, contact_vec, lin_x, lin_y, lin_z, ang_x, ang_y, ang_z,
                          gx, gy, gz, jp, jv, jt, cp):
            state_vec = np.array(state_vec) * state_std + state_mean
            if contact_vec is not None:
                contact_vec = np.array(contact_vec) * contact_std + contact_mean

            lin_x.append(state_vec[0]); lin_y.append(state_vec[1]); lin_z.append(state_vec[2])
            ang_x.append(state_vec[3]); ang_y.append(state_vec[4]); ang_z.append(state_vec[5])
            gx.append(state_vec[6]); gy.append(state_vec[7]); gz.append(state_vec[8])
            for j in JOINT_PICK:
                jp[j].append(state_vec[9 + j])
                jv[j].append(state_vec[off_qd + j])
                jt[j].append(state_vec[off_tau + j])
            if contact_vec is not None:
                for j in range(contact_dims):
                    cp[j].append(contact_vec[j])

        for t in range(M):
            st_t = x[:, t, :state_dims]
            ct_t = x[:, t, state_dims:state_dims+contact_dims]
            state_vec = st_t.squeeze(0).cpu().numpy().tolist()
            contact_vec = ct_t.squeeze(0).cpu().numpy().tolist()
            _append_state(
                state_vec, contact_vec,
                lin_vel_x_pred, lin_vel_y_pred, lin_vel_z_pred,
                ang_vel_x_pred, ang_vel_y_pred, ang_vel_z_pred,
                grav_x_pred, grav_y_pred, grav_z_pred,
                pred_joints, pred_joint_vel, pred_joint_tau, pred_contacts
            )

        with torch.inference_mode():
            for t in range(max_steps-1):
                st_t = torch.unsqueeze(x[:, t, :state_dims], dim=1)
                at_t = torch.unsqueeze(x[:, t, state_dims+contact_dims:state_dims+contact_dims+action_dims], dim=1)
                input_t = torch.cat((st_t, at_t), dim=-1)
                
                if t < M-1:
                    ht = world_model_loaded.forward(input_t, ht, predict=False)           
                else:
                    if t == M-1:
                        x_prev = st_t
                        out = world_model_loaded.forward(input_t, ht, predict=True, sample=True, x_prev=x_prev)
                    else:
                        input_t_pred = torch.cat((st_next_pred, at_t), dim=-1)
                        out = world_model_loaded.forward(input_t_pred, ht, predict=True, sample=True, x_prev=st_next_pred)

                    st_next_pred, ht, contact_pred = out
                    
                    state_vec = torch.squeeze(st_next_pred[-1], dim=0).clone().detach().cpu().numpy().tolist()
                    contact_vec = torch.squeeze(contact_pred[-1], dim=0).clone().detach().cpu().numpy().tolist()
                    _append_state(
                        state_vec, contact_vec,
                        lin_vel_x_pred, lin_vel_y_pred, lin_vel_z_pred,
                        ang_vel_x_pred, ang_vel_y_pred, ang_vel_z_pred,
                        grav_x_pred, grav_y_pred, grav_z_pred,
                        pred_joints, pred_joint_vel, pred_joint_tau, pred_contacts
                    )

        ################## Evaluation ends ##################

        lin_vel_x_true, lin_vel_y_true, lin_vel_z_true = [], [], []
        ang_vel_x_true, ang_vel_y_true, ang_vel_z_true = [], [], []
        grav_x_true, grav_y_true, grav_z_true = [], [], []
        true_joints = {j: [] for j in JOINT_PICK}
        true_joint_vel = {j: [] for j in JOINT_PICK}
        true_joint_tau = {j: [] for j in JOINT_PICK}
        true_contacts = {j: [] for j in range(contact_dims)}

        true_traj = np.squeeze(single_trajectory.numpy())
        for i in range(max_steps):
            transition = true_traj[i, :]
            state = transition[:96].tolist()
            contact = transition[96:126].tolist()
            _append_state(
                state, contact,
                lin_vel_x_true, lin_vel_y_true, lin_vel_z_true,
                ang_vel_x_true, ang_vel_y_true, ang_vel_z_true,
                grav_x_true, grav_y_true, grav_z_true,
                true_joints, true_joint_vel, true_joint_tau, true_contacts
            )

        lin_vel_true = [lin_vel_x_true, lin_vel_y_true, lin_vel_z_true]
        lin_vel_preds = [lin_vel_x_pred, lin_vel_y_pred, lin_vel_z_pred]
        ang_vel_true = [ang_vel_x_true, ang_vel_y_true, ang_vel_z_true]
        ang_vel_preds = [ang_vel_x_pred, ang_vel_y_pred, ang_vel_z_pred]
        grav_true = [grav_x_true, grav_y_true, grav_z_true]
        grav_preds = [grav_x_pred, grav_y_pred, grav_z_pred]

        plot_graphs(lin_vel_true=lin_vel_true,
                    lin_vel_preds=lin_vel_preds,
                    ang_vel_true=ang_vel_true,
                    ang_vel_preds=ang_vel_preds,
                    true_joints=true_joints,
                    pred_joints=pred_joints,
                    M=M,
                    model_dir_name=model_dir_name,
                    model_name=model_name,
                    proj_grav_true=grav_true,
                    proj_grav_pred=grav_preds,
                    true_joint_vel=true_joint_vel,
                    pred_joint_vel=pred_joint_vel,
                    true_joint_tau=true_joint_tau,
                    pred_joint_tau=pred_joint_tau,
                    true_contacts=true_contacts,
                    pred_contacts=pred_contacts)

        




            
            



            