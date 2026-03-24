import json, os, sqlite3
from itertools import chain
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random


class TrajectoryChunksDataset(Dataset):
    def __init__(self, db_path: str = None, M: int = 32, N: int = 8):
        if db_path is None:
            raise ValueError('DB path is not passed')

        self.db_path = db_path
        self.M = M
        self.N = N

        try:        
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM PretrainingData")
            self.rows = cursor.fetchall()
            self.all_trajectories = [json.loads(self.rows[i][0])[f'trajectory-{i+1}'] for i in range(len(self.rows))]
            self.len_of_single_trajectory = len(self.all_trajectories[0])
            self.total_trajectories = len(self.all_trajectories)
            print(f'Db has been successfully loaded. Total number of saved trajectories - {self.total_trajectories}')
            self.all_windows = self.accumalate_windows(M, N, self.all_trajectories)
        except:
            raise ValueError(f'Not able to load db. Does the db exists in the given path - {self.db_path} ?')


    def accumalate_windows(self, M: int, N: int, all_trajectories: list = [], combine_all_traj=True):

        if len(all_trajectories) == 0:
            raise ValueError("No trajectories has been passed. Pass in trajectory to accumalate M+N windows")
        
        all_windows = []
        if not combine_all_traj:
            for trajectory in all_trajectories:
                for i in range(len(trajectory)):
                    tuple_window = trajectory[i : i+(M+N)] # [(s_vec_1, a_vec_1, c_vec_1), (s_vec_2, a_vec_2, c_vec_2), (s_vec_3, a_vec_3, c_vec_3)]
                    if len(tuple_window) == (M+N):
                        window = np.array([list(chain.from_iterable(single_step_tuple[:2] + single_step_tuple[2 + 1:])) for single_step_tuple in tuple_window]) # single sample in the batch is a window of size -> (M+N, s_dim+a_dim)
                        all_windows.append(window)
                print('\n\n')

        else:
            combined_trajectory = list(chain.from_iterable(all_trajectories))
            # print(len(combined_trajectory))
            for i in range(len(combined_trajectory)):
                tuple_window = combined_trajectory[i : i+(M+N)] # [(s_vec_1, a_vec_1, c_vec_1), (s_vec_2, a_vec_2, c_vec_2), (s_vec_3, a_vec_3, c_vec_3)]
                if len(tuple_window) == (M+N):
                    window = np.array([list(chain.from_iterable(single_step_tuple[:1] + single_step_tuple[2:3])) for single_step_tuple in tuple_window], dtype=np.float32)
                    all_windows.append(window)
            # print('\n\n')

        # shuffle and return
        random.shuffle(all_windows)
        return all_windows


    def __len__(self):
        total_windows_per_trajectory = self.len_of_single_trajectory - (self.M + self.N) - 1
        total_windows_across_dataset = self.total_trajectories * total_windows_per_trajectory
        return total_windows_across_dataset


    def __getitem__(self, index):
        return self.all_windows[index]
    
    def get_combined_trajectory(self):
        combined_trajectory = list(chain.from_iterable(self.all_trajectories))
        return combined_trajectory



def load_dataset(db_path: str = None, 
                 batch_size: int = 1024, 
                 single_batch: bool = False, 
                 combine_trajectory: bool = False):
    
    dataset = TrajectoryChunksDataset(db_path=db_path)
    
    if combine_trajectory:
        return dataset.get_combined_trajectory()




    data_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size)
    

    if single_batch:
        for batch in data_loader:
            return batch
        
    

    return data_loader






            


            




        
        
    