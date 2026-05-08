"""Offline evaluation of an MLP world model: pulls one (M + N_pred)-length window
from the same SQLite-backed dataloader used for training, runs the autoregressive
rollout, and plots predicted vs. ground-truth trajectories. No policy/sim required.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from itertools import chain
from pathlib import Path
from typing import List

import numpy as np
import torch

from utils import CreateMlpWMInstance, plot_graphs


def _fetch_eval_window(combined_db_path: str, num_steps: int, mean, std, seed: int = 42):
    """Pull ``num_steps`` consecutive steps from the combined db,
    z-normalize with the provided mean/std, return (num_steps, S+A) float32 array.

    Uses ``seed`` to deterministically pick a starting offset so that both
    MLP and GRU evals select the identical trajectory window by default.

    Bypasses ``load_dataset`` because run_mode="train" eagerly builds every sliding
    window into RAM, which OOMs for large rollout datasets at eval window sizes
    (M + N_pred ~= 152). We only need one window.
    """
    import random as _rng
    conn = sqlite3.connect(combined_db_path)
    try:
        cur = conn.cursor()
        # Fetch all trajectories so we can build the same combined view as GRU eval
        cur.execute("SELECT * FROM PretrainingData")
        rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return None
    # Build a flat list of steps across all trajectories (mirrors GRU's combined_trajectory)
    all_steps = []
    for row in rows:
        blob = json.loads(row[0])
        traj_key = next(iter(blob.keys()))
        all_steps.extend(blob[traj_key])
    if len(all_steps) < num_steps:
        return None
    rng = _rng.Random(seed)
    start_idx = rng.randint(0, len(all_steps) - num_steps)
    steps = all_steps[start_idx:start_idx + num_steps]
    print(f"[eval_mlp] seed={seed}  start_idx={start_idx}  total_steps={len(all_steps)}")
    window = np.array(
        [list(chain.from_iterable(s[:2] + s[2:3])) for s in steps],
        dtype=np.float32,
    )
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, -1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, -1)
    if mean_arr.shape[1] == 125:
        C_mean = np.zeros((1, 30), dtype=np.float32)
        C_std = np.ones((1, 30), dtype=np.float32)
        mean_arr = np.concatenate([mean_arr[:, :96], C_mean, mean_arr[:, 96:]], axis=1)
        std_arr = np.concatenate([std_arr[:, :96], C_std, std_arr[:, 96:]], axis=1)
    window = (window - mean_arr) / (std_arr + 1e-8)
    return window.astype(np.float32), mean_arr, std_arr


def combine_trajectories(transitions_path: str):
    """Mirror of main.combine_trajectories — ensures combined_transitions.db exists."""

    def get_all_rows_per_db(db_path: str):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * From PretrainingData")
        return cursor.fetchall()

    def save_merged_transitions(save_path: str, merged_rows: List, db_name: str = "combined_transitions.db"):
        save_path = os.path.join(save_path, db_name)
        if os.path.isfile(save_path):
            os.remove(save_path)
            print("Removed existing combined db")
        conn = sqlite3.connect(save_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS PretrainingData (trajectories TEXT)")
        cursor.executemany("INSERT INTO PretrainingData (trajectories) VALUES (?)", merged_rows)
        conn.commit()
        print("Saved all trajectories into one db")

    all_transitions_path = os.path.join(os.getcwd(), f"data/{transitions_path}")
    all_transition_dbs = os.listdir(all_transitions_path)
    if "combined_transitions.db" not in all_transition_dbs:
        combined = []
        for f in all_transition_dbs:
            combined += get_all_rows_per_db(os.path.join(all_transitions_path, f))
        save_merged_transitions(all_transitions_path, combined)
    else:
        print("Combined transitions already exists")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline MLP world model evaluation from rollout DB.")
    p.add_argument("--wm-checkpoint", type=Path, required=True,
                   help="Path to MLP world model .pth (state_dict saved by train_mlp).")
    p.add_argument("--db_dir_name", default="pretraining_rollouts",
                   help="Subfolder under data/ containing rollout .db files.")
    p.add_argument("--eval_db_dir_name", default=None,
                   help="Subfolder under data/ containing eval rollout .db files. Defaults to db_dir_name + '_eval'.")
    p.add_argument("--N_pred", type=int, default=None,
                   help="Override config world_model_training_params.N_pred.")
    p.add_argument("--M", type=int, default=None,
                   help="Override config world_model_training_params.M (history-region width on plots).")
    p.add_argument("--wm-model-name", type=str, default="wm_mlp_eval",
                   help="Tag used in plot filenames AND default folder name under logs/plots/.")
    p.add_argument("--wm-model-dir-name", type=str, default=None,
                   help="Subdirectory name under logs/plots/ for saving figures. Defaults to --wm-model-name.")
    p.add_argument("--device", type=str, default=None, help="Override device; default auto.")
    p.add_argument("--all-joints", action="store_true", help="Plot all 29 joints, not the [0,5,20] subset.")
    p.add_argument("--eval-seed", type=int, default=42,
                   help="RNG seed for trajectory selection (default 42). Use the same value for GRU eval to compare on identical data.")
    args = p.parse_args()
    if args.wm_model_dir_name is None:
        args.wm_model_dir_name = args.wm_model_name
    return args


def main() -> None:
    args = parse_args()

    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        print("config file loaded successfully")
    except Exception:
        raise FileNotFoundError("'config.json' not found in the current dir")

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    config["device"] = device

    M_wm = args.M if args.M is not None else config["world_model_training_params"]["M"]
    N_pred = args.N_pred if args.N_pred is not None else config["world_model_training_params"]["N_pred"]
    state_dims = config["robot_params"]["state_dims"]
    action_dims = config["robot_params"]["action_dims"]
    n_joints = action_dims
    expected = 9 + 3 * n_joints
    if state_dims != expected:
        raise ValueError(f"state_dims={state_dims} does not match 9+3*n_joints={expected}")
    contact_dims = config["robot_params"]["contact_dims"]
    off_qd = 9 + n_joints
    off_tau = 9 + 2 * n_joints

    # Build db paths (mirrors main.py)
    eval_db_dir_name = args.eval_db_dir_name if args.eval_db_dir_name else args.db_dir_name + "_eval"
    if os.path.isabs(eval_db_dir_name):
        db_base = eval_db_dir_name
    elif eval_db_dir_name.startswith("data" + os.sep) or eval_db_dir_name == "data":
        db_base = os.path.join(os.getcwd(), eval_db_dir_name)
    else:
        db_base = os.path.join(os.getcwd(), "data", eval_db_dir_name)
    if not os.path.isdir(db_base):
        raise FileNotFoundError(f"Database directory not found: {db_base}")
    db_paths = sorted(os.path.join(db_base, f) for f in os.listdir(db_base) if f.endswith(".db"))
    if not db_paths:
        raise FileNotFoundError(f"No .db files in {db_base}")
    combine_trajectories(eval_db_dir_name)
    combined_db_path = os.path.join(db_base, "combined_transitions.db")

    # Load MLP world model
    world_model = CreateMlpWMInstance(config)
    wm_ckpt_path = args.wm_checkpoint.resolve()
    if not wm_ckpt_path.is_file():
        raise FileNotFoundError(f"WM checkpoint not found: {wm_ckpt_path}")
    payload = torch.load(wm_ckpt_path, map_location=device, weights_only=True)
    if isinstance(payload, dict) and "model" in payload:
        state_dict = payload["model"]
    elif isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload
    world_model.load_state_dict(state_dict, strict=True)
    world_model.eval()
    print(f"loaded MLP world model from {wm_ckpt_path}")

    # Pull one (M + N_pred)-step window directly from the combined DB (z-normed).
    # Avoid load_dataset's run_mode="train" path because it materializes every
    # sliding window in RAM and OOMs on large datasets at this window size.
    window, mean_arr, std_arr = _fetch_eval_window(
        combined_db_path=combined_db_path,
        num_steps=M_wm + N_pred,
        mean=config["mean_state_action"],
        std=config["std_state_action"],
        seed=args.eval_seed,
    )
    if window is None:
        print(f"[WARN] No window of length {M_wm + N_pred} could be sampled from {combined_db_path}.")
        sys.exit(1)

    state_mean = mean_arr[0, :state_dims]
    state_std = std_arr[0, :state_dims]
    contact_mean = mean_arr[0, state_dims:state_dims+contact_dims]
    contact_std = std_arr[0, state_dims:state_dims+contact_dims]
    single_trajectory = torch.from_numpy(window).unsqueeze(0).float()  # (1, M+N_pred, S+A)
    x = single_trajectory.to(device)
    print(f"sampled window shape: {tuple(x.shape)}  (B, M+N_pred, S+A)")

    max_steps = M_wm + N_pred
    JOINT_PICK = list(range(n_joints)) if args.all_joints else [0, 5, 20]

    # ---- Rollout ----
    # Index convention: pred_list[k] holds the model's best estimate of s_k.
    #   k in [0, M-1]: real s_k (warmup region — MLP is stateless, no actual prediction here).
    #   k in [M, M+N_pred-1]: autoregressive prediction.
    # true_list[k] = real s_k for all k. Both arrays have length M+N_pred and are aligned
    # so that pred_list[k] and true_list[k] correspond to the same physical timestep.
    lin_vel_x_pred, lin_vel_y_pred, lin_vel_z_pred = [], [], []
    ang_vel_x_pred, ang_vel_y_pred, ang_vel_z_pred = [], [], []
    grav_x_pred, grav_y_pred, grav_z_pred = [], [], []
    pred_joints = {j: [] for j in JOINT_PICK}
    pred_joint_vel = {j: [] for j in JOINT_PICK}
    pred_joint_tau = {j: [] for j in JOINT_PICK}
    pred_contacts = {j: [] for j in range(contact_dims)}

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

    # Pre-fill the warmup region (k = 0 .. M-1) with real states.
    for k in range(M_wm):
        state_vec = x[0, k, :state_dims].cpu().numpy().tolist()
        contact_vec = x[0, k, state_dims:state_dims+contact_dims].cpu().numpy().tolist()
        _append_state(
            state_vec, contact_vec,
            lin_vel_x_pred, lin_vel_y_pred, lin_vel_z_pred,
            ang_vel_x_pred, ang_vel_y_pred, ang_vel_z_pred,
            grav_x_pred, grav_y_pred, grav_z_pred,
            pred_joints, pred_joint_vel, pred_joint_tau, pred_contacts
        )

    # Autoregressive rollout: predict s_M, s_{M+1}, ..., s_{M+N_pred-1}.
    # Iteration k in [M, M+N_pred-1]: input is (s_{k-1}, a_{k-1}), output is s_k.
    st_next_pred = None
    with torch.inference_mode():
        for k in range(M_wm, M_wm + N_pred):
            if k == M_wm:
                s_in = x[:, k - 1, :state_dims]
                a_in = x[:, k - 1, state_dims+contact_dims:state_dims+contact_dims+action_dims]
            else:
                s_in = st_next_pred
                a_in = x[:, k - 1, state_dims+contact_dims:state_dims+contact_dims+action_dims]
            out = world_model(s_in, a_in, x_prev=s_in)
            if world_model.with_uncertainty:
                st_next_pred, ct_pred, _ = out  # use mean for eval rollout
            else:
                st_next_pred, ct_pred = out
            state_vec = st_next_pred.squeeze(0).cpu().numpy().tolist()
            contact_vec = ct_pred.squeeze(0).cpu().numpy().tolist()
            _append_state(
                state_vec, contact_vec,
                lin_vel_x_pred, lin_vel_y_pred, lin_vel_z_pred,
                ang_vel_x_pred, ang_vel_y_pred, ang_vel_z_pred,
                grav_x_pred, grav_y_pred, grav_z_pred,
                pred_joints, pred_joint_vel, pred_joint_tau, pred_contacts
            )

    # Ground truth: full window (length M + N_pred), aligned 1:1 with pred lists.
    lin_vel_x, lin_vel_y, lin_vel_z = [], [], []
    ang_vel_x, ang_vel_y, ang_vel_z = [], [], []
    grav_x, grav_y, grav_z = [], [], []
    true_joints = {j: [] for j in JOINT_PICK}
    true_joint_vel = {j: [] for j in JOINT_PICK}
    true_joint_tau = {j: [] for j in JOINT_PICK}
    true_contacts = {j: [] for j in range(contact_dims)}
    true_traj = single_trajectory.squeeze(0).cpu().numpy()
    for k in range(max_steps):
        state = true_traj[k, :state_dims].tolist()
        contact = true_traj[k, state_dims:state_dims+contact_dims].tolist()
        _append_state(
            state, contact,
            lin_vel_x, lin_vel_y, lin_vel_z,
            ang_vel_x, ang_vel_y, ang_vel_z,
            grav_x, grav_y, grav_z,
            true_joints, true_joint_vel, true_joint_tau, true_contacts
        )

    plot_graphs(
        lin_vel_true=[lin_vel_x, lin_vel_y, lin_vel_z],
        lin_vel_preds=[lin_vel_x_pred, lin_vel_y_pred, lin_vel_z_pred],
        ang_vel_true=[ang_vel_x, ang_vel_y, ang_vel_z],
        ang_vel_preds=[ang_vel_x_pred, ang_vel_y_pred, ang_vel_z_pred],
        true_joints=true_joints,
        pred_joints=pred_joints,
        M=M_wm,
        model_dir_name=args.wm_model_dir_name,
        model_name=args.wm_model_name,
        not_all_joints=not args.all_joints,
        proj_grav_true=[grav_x, grav_y, grav_z],
        proj_grav_pred=[grav_x_pred, grav_y_pred, grav_z_pred],
        true_joint_vel=true_joint_vel,
        pred_joint_vel=pred_joint_vel,
        true_joint_tau=true_joint_tau,
        pred_joint_tau=pred_joint_tau,
        true_contacts=true_contacts,
        pred_contacts=pred_contacts,
    )
    print("[INFO] eval finished.")


if __name__ == "__main__":
    main()
