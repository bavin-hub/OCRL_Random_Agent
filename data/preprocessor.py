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
                    if "combined" not in db_file:
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


def _extract_trajectory_from_blob(blob: dict):
    if "trajectory-1" in blob:
        return blob["trajectory-1"]
    traj_keys = sorted(k for k in blob.keys() if k.startswith("trajectory-"))
    if not traj_keys:
        raise KeyError(f"No trajectory-* keys found in DB row. Keys: {list(blob.keys())}")
    return blob[traj_keys[0]]


def _extract_done(transition) -> bool:
    """Check if this transition has a done flag (last element, int 1)."""
    if not isinstance(transition, (list, tuple)) or len(transition) == 0:
        return False
    last = transition[-1]
    return isinstance(last, int) and bool(last)


def _find_project_root() -> str:
    candidate = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):
        if os.path.isfile(os.path.join(candidate, "main.py")) and os.path.isdir(os.path.join(candidate, "data")):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    return os.path.abspath(os.path.dirname(__file__))

_PROJECT_ROOT = _find_project_root()

_PATH_KEYS = frozenset(("rgb_path", "depth_path", "rgb_t_path", "depth_t_path",
                         "rgb_t1_path", "depth_t1_path"))

def _resolve_path(p: str) -> str:
    """Resolve a path that may be relative (new format) or absolute (legacy).

    Relative paths are resolved against the project root.
    Absolute paths that don't exist are attempted with the project root prefix
    stripped and re-resolved (handles data moved from another machine).
    """
    if not os.path.isabs(p):
        return os.path.join(_PROJECT_ROOT, p)
    if os.path.exists(p):
        return p
    # Legacy absolute path from a different machine — try to extract the
    # relative portion after the project directory name.
    marker = "OCRL_Random_Agent" + os.sep
    idx = p.find(marker)
    if idx != -1:
        relative = p[idx + len(marker):]
        resolved = os.path.join(_PROJECT_ROOT, relative)
        if os.path.exists(resolved):
            return resolved
    return p

def _extract_rgbd_record(transition):
    if not isinstance(transition, (list, tuple)):
        return None
    if len(transition) < 5:
        return None
    meta = transition[4]
    if not isinstance(meta, dict):
        return None
    meta = {k: (_resolve_path(v) if k in _PATH_KEYS and isinstance(v, str) else v)
            for k, v in meta.items()}
    if ("rgb_t_path" in meta and "depth_t_path" in meta and "rgb_t1_path" in meta and "depth_t1_path" in meta):
        return meta
    if ("rgb_path" in meta and "depth_path" in meta):
        return meta
    return None


def _resize_chw(img: np.ndarray, target_hw: tuple[int, int] | None) -> np.ndarray:
    """Resize a (C, H, W) array to (C, target_h, target_w) using area interpolation."""
    if target_hw is None or (img.shape[1] == target_hw[0] and img.shape[2] == target_hw[1]):
        return img
    import cv2
    C = img.shape[0]
    hwc = np.transpose(img, (1, 2, 0))
    hwc = cv2.resize(hwc, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
    if hwc.ndim == 2:
        hwc = hwc[:, :, None]
    return np.transpose(hwc, (2, 0, 1))


def _as_chw_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[-1] == 3:
        rgb = np.transpose(rgb, (2, 0, 1))
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"Unexpected RGB shape: {rgb.shape}")
    rgb = rgb.astype(np.float32)
    if rgb.max() > 1.0:
        rgb /= 255.0
    return rgb


def _as_chw_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth)
    if depth.ndim == 2:
        depth = depth[None, :, :]
    elif depth.ndim == 3 and depth.shape[-1] == 1:
        depth = np.transpose(depth, (2, 0, 1))
    if depth.ndim != 3 or depth.shape[0] != 1:
        raise ValueError(f"Unexpected depth shape: {depth.shape}")
    return depth.astype(np.float32)


class VisionTransitionDataset(Dataset):
    """Loads seq_len consecutive RGB-D frames from rollout .db files.

    Each item returns (rgb_t, depth_t, rgb_t1, depth_t1, action_t, prop_t)
    as arrays of shape (seq_len, ...).

    Proprioception = state[:38] = lin_vel(3) + ang_vel(3) + gravity(3) + joint_pos(29).
    """

    _PROP_SLICE = slice(0, 38)

    def __init__(self, db_path: str = None, db_paths=None, traj_cache_size: int = 16, seq_len: int = 8,
                 target_size: tuple[int, int] | None = None):
        self.db_paths = _normalize_db_paths(db_path=db_path, db_paths=db_paths)
        self.seq_len = seq_len
        self._target_size = target_size
        self._traj_cache = OrderedDict()
        self._traj_cache_limit = max(1, int(traj_cache_size))
        self._sources = []  # (db_file, rowid)
        self._index = []    # (source_idx, start_step_t)

        for db_file in self.db_paths:
            if "combined" in os.path.basename(db_file):
                continue
            conn = sqlite3.connect(db_file)
            try:
                cur = conn.cursor()
                rows = cur.execute("SELECT rowid, * FROM PretrainingData ORDER BY rowid").fetchall()
                for row in rows:
                    rowid = row[0]
                    traj = _extract_trajectory_from_blob(json.loads(row[1]))
                    source_idx = len(self._sources)
                    self._sources.append((db_file, rowid))

                    # Collect steps that have a valid rgb_path record.
                    valid_steps = set()
                    for i, transition in enumerate(traj):
                        meta = _extract_rgbd_record(transition)
                        if meta is not None and "rgb_path" in meta:
                            valid_steps.add(i)

                    # Index sequences: need seq_len+1 consecutive valid steps.
                    for i in sorted(valid_steps):
                        if all(i + k in valid_steps for k in range(seq_len + 1)):
                            self._index.append((source_idx, i))
            finally:
                conn.close()

        if not self._index:
            raise ValueError("No valid visual sequences found in database(s).")

    def __len__(self):
        return len(self._index)

    def _evict_cache_if_needed(self):
        while len(self._traj_cache) > self._traj_cache_limit:
            self._traj_cache.popitem(last=False)

    def _load_trajectory(self, source_idx: int):
        db_file, rowid = self._sources[source_idx]
        conn = sqlite3.connect(db_file)
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM PretrainingData WHERE rowid = ?", (rowid,))
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"Missing rowid={rowid} in {db_file}")
            return _extract_trajectory_from_blob(json.loads(row[0]))
        finally:
            conn.close()

    def _get_trajectory_cached(self, source_idx: int):
        if source_idx in self._traj_cache:
            self._traj_cache.move_to_end(source_idx)
            return self._traj_cache[source_idx]
        traj = self._load_trajectory(source_idx)
        self._traj_cache[source_idx] = traj
        self._evict_cache_if_needed()
        return traj

    def __getitem__(self, index):
        source_idx, start_t = self._index[int(index)]
        traj = self._get_trajectory_cached(source_idx)

        # Load seq_len+1 frames: start_t to start_t+seq_len (inclusive).
        # Frame k is input at step k; frame k+1 is the prediction target at step k.
        frames_rgb   = []
        frames_depth = []
        for k in range(self.seq_len + 1):
            meta = _extract_rgbd_record(traj[start_t + k])
            rgb = _as_chw_rgb(np.load(meta["rgb_path"]))
            depth = _as_chw_depth(np.load(meta["depth_path"]))
            if self._target_size is not None:
                rgb = _resize_chw(rgb, self._target_size)
                depth = _resize_chw(depth, self._target_size)
            frames_rgb.append(rgb)
            frames_depth.append(depth)

        return {
            "rgb_t":    np.stack(frames_rgb[:-1]),    # (seq_len, 3, H, W)
            "depth_t":  np.stack(frames_depth[:-1]),  # (seq_len, 1, H, W)
            "rgb_t1":   np.stack(frames_rgb[1:]),     # (seq_len, 3, H, W)
            "depth_t1": np.stack(frames_depth[1:]),   # (seq_len, 1, H, W)
            "action_t": np.stack([
                np.asarray(traj[start_t + k][3], dtype=np.float32)
                for k in range(self.seq_len)
            ]),  # (seq_len, action_dim)
            "prop_t": np.stack([
                np.asarray(traj[start_t + k][0], dtype=np.float32)[self._PROP_SLICE]
                for k in range(self.seq_len)
            ]),  # (seq_len, 38)
        }


def load_vision_dataset(
    db_path: str = None,
    db_paths=None,
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = True,
    traj_cache_size: int = 16,
    seq_len: int = 8,
    target_size: tuple[int, int] | None = None,
):
    paths = _normalize_db_paths(db_path=db_path, db_paths=db_paths)
    dataset = VisionTransitionDataset(db_paths=paths, traj_cache_size=traj_cache_size, seq_len=seq_len,
                                      target_size=target_size)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
