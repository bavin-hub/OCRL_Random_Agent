import json, os, sqlite3
from itertools import chain
from collections import OrderedDict
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random


def _normalize_db_paths(db_path=None, db_paths=None):
    if db_paths is not None:
        paths = list(db_paths)
    elif db_path is not None:
        paths = [db_path]
    else:
        raise ValueError('Pass db_path or db_paths')
    paths = [os.path.abspath(p) for p in paths]
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f'Rollout database not found: {p}')
    return paths


class LazyCombinedTrajectoryView:
    """Sliceable view over the flattened rollout stream without loading all steps into RAM."""

    def __init__(self, dataset):
        self._ds = dataset

    def __len__(self):
        return self._ds.combined_len

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                return [self._ds._get_combined_step(i) for i in range(start, stop, step)]
            return [self._ds._get_combined_step(i) for i in range(start, stop)]
        return self._ds._get_combined_step(idx)


class TrajectoryChunksDataset(Dataset):
    def __init__(self, db_path: str = None, db_paths=None, M: int = 32, N: int = 8, traj_cache_size: int = 32, 
                 run_mode: str = None, combined_db_path: str = None):
        paths = _normalize_db_paths(db_path=db_path, db_paths=db_paths)
        self.db_paths = paths
        self.M = M
        self.N = N
        self._traj_cache = OrderedDict()
        self._traj_cache_limit = max(1, int(traj_cache_size))
        self.run_mode = run_mode
        self.combined_db_path = combined_db_path

        self._sources = []
        traj_lengths = []

        try:
            if self.run_mode == "train":
                # pass
                conn = sqlite3.connect(self.combined_db_path)
                cur = conn.cursor()
                query = '''SELECT * From PretrainingData'''
                cur.execute(query)

                rows = cur.fetchall()
                print('total rows : ', len(rows))

                all_trajectories = [json.loads(traj[0])['trajectory-1'] for i, traj in enumerate(rows)]
                # print(type(all_trajectories))
                # print(len(all_trajectories))
                # print(type(all_trajectories[0]))
                # print(type(all_trajectories[0][0]))
                # print(len(all_trajectories[0][0][0]))

                # print(all_trajectories)
                self.all_windows = self.accumalate_windows(self.M,
                                                           self.N,
                                                           all_trajectories=all_trajectories)


            else:
                for db_file in self.db_paths:
                    conn = sqlite3.connect(db_file)
                    cur = conn.cursor()
                    for file_k, row in enumerate(cur.execute('SELECT rowid, * FROM PretrainingData ORDER BY rowid')):
                        rowid = row[0]
                        payload = row[1]
                        blob = json.loads(payload)
                        traj = blob[f'trajectory-{file_k + 1}']
                        traj_lengths.append(len(traj))
                        del traj
                        self._sources.append((db_file, rowid, file_k))
                    conn.close()

                if not traj_lengths:
                    raise ValueError('No trajectories found in database(s)')

                self._cum_lens = np.zeros(len(traj_lengths) + 1, dtype=np.int64)
                self._cum_lens[1:] = np.cumsum(traj_lengths)
                self.combined_len = int(self._cum_lens[-1])
                self.len_of_single_trajectory = traj_lengths[0]
                self.total_trajectories = len(traj_lengths)
                print(
                    f'Db index built. Shards: {len(self.db_paths)}, '
                    f'trajectories: {self.total_trajectories}, combined steps: {self.combined_len}'
                )
            # pass
        except Exception as e:
            paths_str = ', '.join(self.db_paths)
            raise ValueError(f'Not able to load rollout DB(s) at [{paths_str}]: {e}') from e

    def _evict_cache_if_needed(self):
        while len(self._traj_cache) > self._traj_cache_limit:
            self._traj_cache.popitem(last=False)

    def _load_trajectory(self, traj_idx: int):
        db_file, rowid, file_k = self._sources[traj_idx]
        conn = sqlite3.connect(db_file)
        try:
            cur = conn.cursor()
            cur.execute('SELECT * FROM PretrainingData WHERE rowid = ?', (rowid,))
            row = cur.fetchone()
            if row is None:
                raise ValueError(f'Missing rowid={rowid} in {db_file}')
            payload = row[0]
            blob = json.loads(payload)
            return blob[f'trajectory-{file_k + 1}']
        finally:
            conn.close()

    def _get_trajectory_cached(self, traj_idx: int):
        if traj_idx in self._traj_cache:
            self._traj_cache.move_to_end(traj_idx)
            return self._traj_cache[traj_idx]
        traj = self._load_trajectory(traj_idx)
        self._traj_cache[traj_idx] = traj
        self._evict_cache_if_needed()
        return traj

    def _global_step_to_traj_and_offset(self, global_step: int):
        traj_idx = int(np.searchsorted(self._cum_lens, global_step, side='right') - 1)
        offset = int(global_step - self._cum_lens[traj_idx])
        return traj_idx, offset

    def _get_combined_step(self, global_step: int):
        if global_step < 0 or global_step >= self.combined_len:
            raise IndexError(global_step)
        traj_idx, offset = self._global_step_to_traj_and_offset(global_step)
        traj = self._get_trajectory_cached(traj_idx)
        return traj[offset]

    @property
    def combined_trajectory(self):
        return LazyCombinedTrajectoryView(self)

    def accumalate_windows(self, M: int, N: int, all_trajectories: list = [], combine_all_traj=True):
        if len(all_trajectories) == 0:
            raise ValueError('No trajectories has been passed. Pass in trajectory to accumalate M+N windows')

        all_windows = []
        if not combine_all_traj:
            for trajectory in all_trajectories:
                for i in range(len(trajectory)):
                    tuple_window = trajectory[i : i + (M + N)]
                    if len(tuple_window) == (M + N):
                        window = np.array(
                            [
                                list(chain.from_iterable(single_step_tuple[:2] + single_step_tuple[2 + 1 :]))
                                for single_step_tuple in tuple_window
                            ]
                        )
                        all_windows.append(window)

        else:
            print('good inside the else part')
            combined_trajectory = list(chain.from_iterable(all_trajectories))
            for i in range(len(combined_trajectory)):
                tuple_window = combined_trajectory[i : i + (M + N)]
                if len(tuple_window) == (M + N):
                    window = np.array(
                        [
                            list(chain.from_iterable(single_step_tuple[:1] + single_step_tuple[2:3]))
                            for single_step_tuple in tuple_window
                        ],
                        dtype=np.float32,
                    )
                    all_windows.append(window)

        random.shuffle(all_windows)
        return all_windows

    def __len__(self):
        if self.run_mode == "train":
            return len(self.all_windows)
        return max(0, self.combined_len - (self.M + self.N) + 1)

    def __getitem__(self, index):
        if self.run_mode == "train":
            return self.all_windows[index]
        start = int(index)
        steps = [self._get_combined_step(start + j) for j in range(self.M + self.N)]
        return np.array(
            [list(chain.from_iterable(s[:1] + s[2:3])) for s in steps],
            dtype=np.float32,
        )

    def get_combined_trajectory(self):
        return self.combined_trajectory


def load_dataset(
    db_path: str = None,
    db_paths=None,
    batch_size: int = 1024,
    single_batch: bool = False,
    combine_trajectory: bool = False,
    M: int = 32,
    N: int = 8,
    traj_cache_size: int = 32,
    combined_db_path: str = None,
    run_mode: str = None
):
    paths = _normalize_db_paths(db_path=db_path, db_paths=db_paths)
    dataset = TrajectoryChunksDataset(db_paths=paths, M=M, N=N, traj_cache_size=traj_cache_size, 
                                      combined_db_path=combined_db_path, run_mode=run_mode)

    if combine_trajectory:
        return dataset.get_combined_trajectory()

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size)

    if single_batch:
        for batch in data_loader:
            return batch

    return data_loader
