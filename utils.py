# contains plot, save, load utils 
import os, json
import numpy as np
import torch
from models.world_model import RandomWorldStepGru
import matplotlib.pyplot as plt
from datetime import datetime

def get_model_name(model_type):
    string_date_time = "_".join(str(datetime.now()).split(".")[0].split(" "))
    model_name = model_type + "_" + string_date_time
    return model_name

def CreateWorlModelInstance(config):
    world_model = RandomWorldStepGru(batch_size=config['world_model_training_params']['batch_size'],
                                    state_dim=config['robot_params']['state_dims'],
                                    contact_dim=config['robot_params']['contact_dims'],
                                    action_dim=config['robot_params']['action_dims'],
                                    embed_dim=config['world_model_arch_params']['embed_dim'],
                                    hidden_dim=config['world_model_arch_params']['gru_hidden_dim'],
                                    num_gru_layers=config['world_model_arch_params']['num_gru_layers'],
                                    mlp_dim=config['world_model_arch_params']['mlp_head_dim'],
                                    lr=config['world_model_training_params']['learning_rate'],
                                    weight_decay=config['world_model_training_params']['weight_decay'],
                                    device=config['device'])

    return world_model



def create_runs_dir(model_dir_name):
    models_dir = os.path.join(os.path.join(os.getcwd(), f'logs/saved_models/{model_dir_name}'))
    if not os.path.isdir(models_dir):
        print('Creating runs directory')
        os.makedirs(models_dir)

    plots_dir = os.path.join(os.path.join(os.getcwd(), f'logs/plots/{model_dir_name}'))
    if not os.path.isdir(plots_dir):
        print('Creating plots directory')
        os.makedirs(plots_dir)

    return models_dir, plots_dir



def SaveModel(model_obj, model_name, model_dir_name):
    save_dir, _ = create_runs_dir(model_dir_name)
    model_dir = os.path.join(save_dir, model_name)
    torch.save(model_obj.state_dict(), model_dir)
    print("\n#########")
    print("Model saved")
    print("#########\n")


def LoadModel(model_obj, model_name, model_dir_name):
    load_dir, _ = create_runs_dir(model_dir_name)
    model_dir = os.path.join(load_dir, model_name)
    model_obj.load_state_dict(torch.load(model_dir, weights_only=True))
    print("\n#########")
    print("Model Loaded")
    print("#########\n")
    return model_obj


def save_plot_figure(fig, model_dir_name: str, filename: str, dpi: int = 150):
    """Write ``fig`` to ``logs/plots/{model_dir_name}/{filename}`` (adds .png if no extension)."""
    _, plots_dir = create_runs_dir(model_dir_name)
    if not any(filename.lower().endswith(ext) for ext in (".png", ".pdf", ".svg")):
        filename = f"{filename}.png"
    out_path = os.path.join(plots_dir, filename)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot: {out_path}")


def plot_graphs(lin_vel_true, lin_vel_preds, ang_vel_true, 
                ang_vel_preds, true_joints, pred_joints, 
                M, model_dir_name, model_name, not_all_joints=True):
    
    lin_vel_x, lin_vel_y, lin_vel_z = lin_vel_true
    lin_vel_x_pred, lin_vel_y_pred, lin_vel_z_pred = lin_vel_preds

    ang_vel_x, ang_vel_y, ang_vel_z = ang_vel_true
    ang_vel_x_pred, ang_vel_y_pred, ang_vel_z_pred = ang_vel_preds

    if not_all_joints:
        JOINT_PICK = [0, 5, 20]
    else:
        JOINT_PICK = [i for i in range(29)]

    n = len(lin_vel_x)
    time_steps = list(range(1, n + 1))

    c_true = "#356CD2"
    c_pred = "#CF3434"
    c_mark = "#049009"
    lw = 3.5

    def mark_history(ax):
        ax.axvline(M, color=c_mark, linestyle="--", linewidth=1.8, label=f"M={M} (history)")
        ax.legend(loc="best", fontsize=8)

    fig_lin, axes_lin = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
    lin_true = [lin_vel_x, lin_vel_y, lin_vel_z]
    lin_pred = [lin_vel_x_pred, lin_vel_y_pred, lin_vel_z_pred]
    labels_lin = ["base_lin_vel_x", "base_lin_vel_y", "base_lin_vel_z"]
    for k, ax in enumerate(axes_lin):
        ax.plot(time_steps, lin_true[k], label="true", color=c_true, linewidth=lw)
        ax.plot(time_steps, lin_pred[k], label="pred", linestyle=":", color=c_pred, linewidth=lw)
        ax.set_ylabel(labels_lin[k])
        mark_history(ax)
    axes_lin[-1].set_xlabel("time step (1-based)")
    fig_lin.suptitle("Base linear velocity")
    fig_lin.tight_layout()
    save_plot_figure(fig_lin, model_dir_name, f"{model_name}_base_lin_vel")
    plt.show()

    fig_ang, axes_ang = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
    ang_true = [ang_vel_x, ang_vel_y, ang_vel_z]
    ang_pred = [ang_vel_x_pred, ang_vel_y_pred, ang_vel_z_pred]
    labels_ang = ["base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z"]
    for k, ax in enumerate(axes_ang):
        ax.plot(time_steps, ang_true[k], label="true", color=c_true, linewidth=lw)
        ax.plot(time_steps, ang_pred[k], label="pred", linestyle=":", color=c_pred, linewidth=lw)
        ax.set_ylabel(labels_ang[k])
        mark_history(ax)
    axes_ang[-1].set_xlabel("time step (1-based)")
    fig_ang.suptitle("Base angular velocity")
    fig_ang.tight_layout()
    save_plot_figure(fig_ang, model_dir_name, f"{model_name}_base_ang_vel")
    plt.show()

    fig_j, axes_j = plt.subplots(len(JOINT_PICK), 1, figsize=(20, 5 * len(JOINT_PICK)), sharex=True)
    if len(JOINT_PICK) == 1:
        axes_j = [axes_j]
    for ax, j in zip(axes_j, JOINT_PICK):
        ax.plot(time_steps, true_joints[j], label="true", color=c_true, linewidth=lw)
        ax.plot(time_steps, pred_joints[j], label="pred", linestyle=":", color=c_pred, linewidth=lw)
        ax.set_ylabel(f"joint {j}")
        mark_history(ax)
    axes_j[-1].set_xlabel("time step (1-based)")
    fig_j.suptitle("Joint positions (subset)")
    fig_j.tight_layout()
    save_plot_figure(fig_j, model_dir_name, f"{model_name}_joints")
    plt.show()





    