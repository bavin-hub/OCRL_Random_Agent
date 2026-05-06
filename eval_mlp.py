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


def _fetch_eval_window(combined_db_path: str, num_steps: int, mean, std):
    """Pull ``num_steps`` consecutive steps from the first trajectory in combined db,
    z-normalize with the provided mean/std, return (num_steps, S+A) float32 array.

    Bypasses ``load_dataset`` because run_mode="train" eagerly builds every sliding
    window into RAM, which OOMs for large rollout datasets at eval window sizes
    (M + N_pred ~= 152). We only need one window.
    """
    conn = sqlite3.connect(combined_db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM PretrainingData LIMIT 1")
        row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    blob = json.loads(row[0])
    traj_key = next(iter(blob.keys()))
    traj = blob[traj_key]
    if len(traj) < num_steps:
        return None
    steps = traj[:num_steps]
    window = np.array(
        [list(chain.from_iterable(s[:1] + s[2:3])) for s in steps],
        dtype=np.float32,
    )
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, -1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, -1)
    window = (window - mean_arr) / (std_arr + 1e-8)
    return window.astype(np.float32)


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
    p.add_argument("--N_pred", type=int, default=None,
                   help="Override config world_model_training_params.N_pred.")
    p.add_argument("--M", type=int, default=None,
                   help="Override config world_model_training_params.M (history-region width on plots).")
    p.add_argument("--wm-model-name", type=str, default="wm_mlp_eval",
                   help="Tag used in plot filenames.")
    p.add_argument("--wm-model-dir-name", type=str, default="wm_mlp_eval",
                   help="Subdirectory name under logs/plots/ for saving figures.")
    p.add_argument("--device", type=str, default=None, help="Override device; default auto.")
    p.add_argument("--all-joints", action="store_true", help="Plot all 29 joints, not the [0,5,20] subset.")
    return p.parse_args()


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
    off_qd = 9 + n_joints
    off_tau = 9 + 2 * n_joints

    # Build db paths (mirrors main.py)
    if os.path.isabs(args.db_dir_name):
        db_base = args.db_dir_name
    elif args.db_dir_name.startswith("data" + os.sep) or args.db_dir_name == "data":
        db_base = os.path.join(os.getcwd(), args.db_dir_name)
    else:
        db_base = os.path.join(os.getcwd(), "data", args.db_dir_name)
    if not os.path.isdir(db_base):
        raise FileNotFoundError(f"Database directory not found: {db_base}")
    db_paths = sorted(os.path.join(db_base, f) for f in os.listdir(db_base) if f.endswith(".db"))
    if not db_paths:
        raise FileNotFoundError(f"No .db files in {db_base}")
    combine_trajectories(args.db_dir_name)
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
    window = _fetch_eval_window(
        combined_db_path=combined_db_path,
        num_steps=M_wm + N_pred,
        mean=config["mean_state_action"],
        std=config["std_state_action"],
    )
    if window is None:
        print(f"[WARN] No window of length {M_wm + N_pred} could be sampled from {combined_db_path}.")
        sys.exit(1)
    single_trajectory = torch.from_numpy(window).unsqueeze(0).float()  # (1, M+N_pred, S+A)
    x = single_trajectory.to(device)
    print(f"sampled window shape: {tuple(x.shape)}  (B, M+N_pred, S+A)")

    max_steps = M_wm + N_pred
    JOINT_PICK = list(range(n_joints)) if args.all_joints else [0, 5, 20]

    # ---- Rollout ----
    # NOTE: t < M_wm-1 is a fake warmup region for the MLP — the model is stateless,
    # so we just plot ground-truth on the predicted line there. Actual prediction
    # starts at t == M_wm - 1 and runs autoregressively for N_pred steps.
    lin_vel_x_pred, lin_vel_y_pred, lin_vel_z_pred = [], [], []
    ang_vel_x_pred, ang_vel_y_pred, ang_vel_z_pred = [], [], []
    grav_x_pred, grav_y_pred, grav_z_pred = [], [], []
    pred_joints = {j: [] for j in JOINT_PICK}
    pred_joint_vel = {j: [] for j in JOINT_PICK}
    pred_joint_tau = {j: [] for j in JOINT_PICK}

    st_next_pred = None
    with torch.inference_mode():
        for t in range(max_steps - 1):
            if t < M_wm - 1:
                state_vec = x[0, t, :state_dims].cpu().numpy().tolist()
            else:
                if t == M_wm - 1:
                    s_in = x[:, t, :state_dims]
                    a_in = x[:, t, state_dims:state_dims + action_dims]
                    x_prev = s_in
                else:
                    s_in = st_next_pred
                    a_in = x[:, t, -action_dims:]
                    x_prev = st_next_pred
                st_next_pred = world_model(s_in, a_in, x_prev=x_prev)
                state_vec = st_next_pred.squeeze(0).cpu().numpy().tolist()

            lin_vel_x_pred.append(state_vec[0])
            lin_vel_y_pred.append(state_vec[1])
            lin_vel_z_pred.append(state_vec[2])
            ang_vel_x_pred.append(state_vec[3])
            ang_vel_y_pred.append(state_vec[4])
            ang_vel_z_pred.append(state_vec[5])
            grav_x_pred.append(state_vec[6])
            grav_y_pred.append(state_vec[7])
            grav_z_pred.append(state_vec[8])
            for j in JOINT_PICK:
                pred_joints[j].append(state_vec[9 + j])
                pred_joint_vel[j].append(state_vec[off_qd + j])
                pred_joint_tau[j].append(state_vec[off_tau + j])

    # Ground truth
    lin_vel_x, lin_vel_y, lin_vel_z = [], [], []
    ang_vel_x, ang_vel_y, ang_vel_z = [], [], []
    grav_x, grav_y, grav_z = [], [], []
    true_joints = {j: [] for j in JOINT_PICK}
    true_joint_vel = {j: [] for j in JOINT_PICK}
    true_joint_tau = {j: [] for j in JOINT_PICK}
    true_traj = single_trajectory.squeeze(0).cpu().numpy()  # (M+N_pred, S+A)
    for i in range(max_steps - 1):
        state = true_traj[i, :state_dims].tolist()
        lin_vel_x.append(state[0]); lin_vel_y.append(state[1]); lin_vel_z.append(state[2])
        ang_vel_x.append(state[3]); ang_vel_y.append(state[4]); ang_vel_z.append(state[5])
        grav_x.append(state[6]); grav_y.append(state[7]); grav_z.append(state[8])
        for j in JOINT_PICK:
            true_joints[j].append(state[9 + j])
            true_joint_vel[j].append(state[off_qd + j])
            true_joint_tau[j].append(state[off_tau + j])

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
    )
    print("[INFO] eval finished.")


if __name__ == "__main__":
    main()
