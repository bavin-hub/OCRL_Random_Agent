from models.world_model import RandomWorldStepGru
# contains plot, save, load utils 
import os, json
import numpy as np
import torch
#from models.world_model import RandomWorldStepGru
from wm import TransWM, MlpWM
import matplotlib.pyplot as plt
from datetime import datetime

def get_model_name(model_type):
    string_date_time = "_".join(str(datetime.now()).split(".")[0].split(" "))
    model_name = model_type + "_" + string_date_time
    return model_name

def CreateWorlModelInstance(config):

    world_model = TransWM(
    state_dim=config['robot_params']['state_dims'],
    action_dim=config['robot_params']['action_dims'],

    embed_dim=config['world_model_arch_params']['embed_dim'],
    mlp_dim=config['world_model_arch_params']['mlp_dim'],
    num_heads=config['world_model_arch_params'].get('num_heads', 4),
    num_layers=config['world_model_arch_params'].get('num_layers', 1),

    lr=config['world_model_training_params']['learning_rate'],
    weight_decay=config['world_model_training_params']['weight_decay'],

    std_range=(0.01, 0.06),
    dropout=config['world_model_arch_params'].get('dropout', 0.0),

    device=config['device']
    )


    return world_model


def CreateMlpWMInstance(config):
    arch = config['world_model_arch_params_mlp']
    world_model = MlpWM(
        state_dim=config['robot_params']['state_dims'],
        action_dim=config['robot_params']['action_dims'],
        hidden_dim=arch['hidden_dim'],
        num_layers=arch['num_layers'],
        mlp_head_dim=arch['mlp_head_dim'],
        with_uncertainty=arch.get('with_uncertainty', False),
        lr=arch.get('lr', config['world_model_training_params'].get('learning_rate', 1e-3)),
        weight_decay=config['world_model_training_params'].get('weight_decay', 0.0),
        device=config['device'],
    )
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

    ckpts_dir = os.path.join(os.path.join(os.getcwd(), f"logs/ckpts/{model_dir_name}"))
    if not os.path.isdir(ckpts_dir):
        print("Creating ckpts dir")
        os.makedirs(ckpts_dir)

    return models_dir, plots_dir, ckpts_dir



def SaveModel(model_obj, model_name, model_dir_name):
    save_dir, _, _ = create_runs_dir(model_dir_name)
    model_dir = os.path.join(save_dir, model_name)
    torch.save(model_obj.state_dict(), model_dir)
    print("\n#########")
    print("Model saved")
    print("#########\n")


def LoadModel(model_obj, model_name, model_dir_name):
    load_dir, _, _ = create_runs_dir(model_dir_name)
    model_dir = os.path.join(load_dir, model_name)
    model_obj.load_state_dict(torch.load(model_dir, weights_only=True))
    print("\n#########")
    print("Model Loaded")
    print("#########\n")
    return model_obj


def SaveCkpt(model_obj, model_name, model_dir_name, current_epoch, current_loss):
    _, _, ckpt_dir = create_runs_dir(model_dir_name)
    ckpt_dir = os.path.join(ckpt_dir, model_name)

    checkpoint = {"model_state_dict": model_obj.state_dict(),
                  "optimizer_state_dict": model_obj.optimizer.state_dict(),
                  "epoch":current_epoch,
                  "loss":current_loss}
    torch.save(checkpoint, ckpt_dir)
    print("\n\n#########")
    print("Checkpoint saved")
    print("#########\n")


def LoadCkpt(model_obj, model_name, model_dir_name):
    _, _, ckpt_dir = create_runs_dir(model_dir_name)
    ckpt_dir = os.path.join(ckpt_dir, model_name)

    checkpoints = torch.load(ckpt_dir)

    model_obj.load_state_dict(checkpoints["model_state_dict"])
    model_obj.optimizer.load_state_dict(checkpoints["optimizer_state_dict"])
    model_obj.train()
    print("\n\n#########")
    print("Checkpoint loaded")
    print("#########\n")
    return model_obj





def count_parameters(model):
    trainable_params = 0
    non_trainable_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable_params += parameter.numel()
        else:
            non_trainable_params += parameter.numel()
            
    print(f"Total Trainable Params: {trainable_params}")
    print(f"Total Non-Trainable Params: {non_trainable_params}")
    print(f"Total Params: {trainable_params + non_trainable_params}")
    return trainable_params, non_trainable_params



def save_plot_figure(fig, model_dir_name: str, filename: str, dpi: int = 150):
    """Write ``fig`` to ``logs/plots/{model_dir_name}/{filename}`` (adds .png if no extension)."""
    _, plots_dir, _ = create_runs_dir(model_dir_name)
    if not any(filename.lower().endswith(ext) for ext in (".png", ".pdf", ".svg")):
        filename = f"{filename}.png"
    out_path = os.path.join(plots_dir, filename)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot: {out_path}")


def plot_graphs(
    lin_vel_true,
    lin_vel_preds,
    ang_vel_true,
    ang_vel_preds,
    true_joints,
    pred_joints,
    M,
    model_dir_name,
    model_name,
    not_all_joints=True,
    proj_grav_true=None,
    proj_grav_pred=None,
    true_joint_vel=None,
    pred_joint_vel=None,
    true_joint_tau=None,
    pred_joint_tau=None,
    true_contacts=None,
    pred_contacts=None,
):
    """
    State vector layout when extended plots are used:
    [base_lin_vel(3), base_ang_vel(3), proj_gravity(3), joint_pos(n), joint_vel(n), joint_torques(n)].
    Optional *_grav / *_joint_vel / *_joint_tau args are omitted in callers that only log the legacy fields.
    """
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

    def _rollout_rmse(true_lists, pred_lists):
        """RMSE over the rollout region (k >= M). pred_list[k] and true_list[k] both hold
        the value at physical timestep k, so the comparison is direct."""
        diffs = []
        for tl, pl in zip(true_lists, pred_lists):
            if len(pl) <= M or len(tl) <= M:
                continue
            p = np.asarray(pl[M:], dtype=np.float64)
            t = np.asarray(tl[M:], dtype=np.float64)
            n = min(len(p), len(t))
            if n == 0:
                continue
            diffs.append((p[:n] - t[:n]) ** 2)
        if not diffs:
            return float("nan")
        return float(np.sqrt(np.mean(np.concatenate(diffs))))

    def plot_joint_dict_figure(true_dict, pred_dict, title: str, fname_suffix: str, pick_list=None):
        if pick_list is None:
            pick_list = JOINT_PICK
        rmse = _rollout_rmse(
            [true_dict[j] for j in pick_list],
            [pred_dict[j] for j in pick_list],
        )
        fig_j, axes_j = plt.subplots(
            len(pick_list), 1, figsize=(20, 5 * len(pick_list)), sharex=True
        )
        if len(pick_list) == 1:
            axes_j = [axes_j]
        for ax, j in zip(axes_j, pick_list):
            ax.plot(time_steps, true_dict[j], label="true", color=c_true, linewidth=lw)
            ax.plot(time_steps, pred_dict[j], label="pred", linestyle=":", color=c_pred, linewidth=lw)
            ax.set_ylabel(f"joint {j}")
            mark_history(ax)
        axes_j[-1].set_xlabel("time step (1-based)")
        fig_j.suptitle(f"{title}  |  rollout RMSE (norm) = {rmse:.5f}")
        fig_j.tight_layout()
        save_plot_figure(fig_j, model_dir_name, f"{model_name}_{fname_suffix}")
        plt.show()
        return rmse

    metrics = {}

    lin_true = [lin_vel_x, lin_vel_y, lin_vel_z]
    lin_pred = [lin_vel_x_pred, lin_vel_y_pred, lin_vel_z_pred]
    rmse_lin = _rollout_rmse(lin_true, lin_pred)
    metrics["base_lin_vel_rmse"] = rmse_lin
    fig_lin, axes_lin = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
    labels_lin = ["base_lin_vel_x", "base_lin_vel_y", "base_lin_vel_z"]
    for k, ax in enumerate(axes_lin):
        ax.plot(time_steps, lin_true[k], label="true", color=c_true, linewidth=lw)
        ax.plot(time_steps, lin_pred[k], label="pred", linestyle=":", color=c_pred, linewidth=lw)
        ax.set_ylabel(labels_lin[k])
        mark_history(ax)
    axes_lin[-1].set_xlabel("time step (1-based)")
    fig_lin.suptitle(f"Base linear velocity  |  rollout RMSE (norm) = {rmse_lin:.5f}")
    fig_lin.tight_layout()
    save_plot_figure(fig_lin, model_dir_name, f"{model_name}_base_lin_vel")
    plt.show()

    ang_true = [ang_vel_x, ang_vel_y, ang_vel_z]
    ang_pred = [ang_vel_x_pred, ang_vel_y_pred, ang_vel_z_pred]
    rmse_ang = _rollout_rmse(ang_true, ang_pred)
    metrics["base_ang_vel_rmse"] = rmse_ang
    fig_ang, axes_ang = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
    labels_ang = ["base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z"]
    for k, ax in enumerate(axes_ang):
        ax.plot(time_steps, ang_true[k], label="true", color=c_true, linewidth=lw)
        ax.plot(time_steps, ang_pred[k], label="pred", linestyle=":", color=c_pred, linewidth=lw)
        ax.set_ylabel(labels_ang[k])
        mark_history(ax)
    axes_ang[-1].set_xlabel("time step (1-based)")
    fig_ang.suptitle(f"Base angular velocity  |  rollout RMSE (norm) = {rmse_ang:.5f}")
    fig_ang.tight_layout()
    save_plot_figure(fig_ang, model_dir_name, f"{model_name}_base_ang_vel")
    plt.show()

    rmse_joints = _rollout_rmse(
        [true_joints[j] for j in JOINT_PICK],
        [pred_joints[j] for j in JOINT_PICK],
    )
    metrics["joint_pos_rmse"] = rmse_joints
    fig_j, axes_j = plt.subplots(len(JOINT_PICK), 1, figsize=(20, 5 * len(JOINT_PICK)), sharex=True)
    if len(JOINT_PICK) == 1:
        axes_j = [axes_j]
    for ax, j in zip(axes_j, JOINT_PICK):
        ax.plot(time_steps, true_joints[j], label="true", color=c_true, linewidth=lw)
        ax.plot(time_steps, pred_joints[j], label="pred", linestyle=":", color=c_pred, linewidth=lw)
        ax.set_ylabel(f"joint {j}")
        mark_history(ax)
    axes_j[-1].set_xlabel("time step (1-based)")
    fig_j.suptitle(f"Joint positions (subset)  |  rollout RMSE (norm) = {rmse_joints:.5f}")
    fig_j.tight_layout()
    save_plot_figure(fig_j, model_dir_name, f"{model_name}_joints")
    plt.show()

    if proj_grav_true is not None and proj_grav_pred is not None:
        gx_t, gy_t, gz_t = proj_grav_true
        gx_p, gy_p, gz_p = proj_grav_pred
        grav_true = [gx_t, gy_t, gz_t]
        grav_pred = [gx_p, gy_p, gz_p]
        mse_grav = _rollout_rmse(grav_true, grav_pred)
        metrics["proj_gravity_rmse"] = mse_grav
        fig_g, axes_g = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
        labels_g = ["proj_gravity_x", "proj_gravity_y", "proj_gravity_z"]
        for k, ax in enumerate(axes_g):
            ax.plot(time_steps, grav_true[k], label="true", color=c_true, linewidth=lw)
            ax.plot(time_steps, grav_pred[k], label="pred", linestyle=":", color=c_pred, linewidth=lw)
            ax.set_ylabel(labels_g[k])
            mark_history(ax)
        axes_g[-1].set_xlabel("time step (1-based)")
        fig_g.suptitle(f"Projected gravity (body frame)  |  rollout RMSE (norm) = {mse_grav:.5f}")
        fig_g.tight_layout()
        save_plot_figure(fig_g, model_dir_name, f"{model_name}_proj_gravity")
        plt.show()

    if true_joint_vel is not None and pred_joint_vel is not None:
        metrics["joint_vel_rmse"] = plot_joint_dict_figure(
            true_joint_vel, pred_joint_vel, "Joint velocities (subset)", "joint_vel"
        )

    if true_joint_tau is not None and pred_joint_tau is not None:
        metrics["joint_tau_rmse"] = plot_joint_dict_figure(
            true_joint_tau, pred_joint_tau, "Joint torques (subset)", "joint_torques"
        )

    if true_contacts is not None and pred_contacts is not None:
        # Pick a few body contacts to plot
        BODY_CONTACT_PICK = [0, 8, 14] if not_all_joints else list(range(26))
        metrics["contact_rmse"] = plot_joint_dict_figure(
            true_contacts, pred_contacts, "Body Contacts (subset)", "body_contacts", pick_list=BODY_CONTACT_PICK
        )
        
        # Explicitly plot the foot contacts / forces (last 4 indices)
        FOOT_CONTACT_PICK = [26, 27, 28, 29]
        metrics["foot_contact_rmse"] = plot_joint_dict_figure(
            true_contacts, pred_contacts, "Foot Contacts & Forces", "foot_contacts", pick_list=FOOT_CONTACT_PICK
        )

    _, plots_dir, _ = create_runs_dir(model_dir_name)
    metrics_path = os.path.join(plots_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")
    print("Rollout RRMSE (physical space):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")


def z_norm(state_action_pair, mean, std):
    # print(type(state_action_pair))
    state_action_pair = (state_action_pair - mean) / (std + 1e-8)
    return state_action_pair.astype(np.float32)


def minmax_norm_state_action_pair(
    state_action_pair,
    joint_pos_min,
    joint_pos_max,
    tau_min,
    tau_max,
    *,
    st_dim: int = 96,
    base_lin_vel_limit: float = 4.0,
    base_ang_vel_limit: float = 10.0,
    gravity_limit: float = 9.81,
    joint_vel_limit: float = 15.0,
):
    """Normalize a concatenated [st | at] vector to [-1, 1] component-wise.

    Last axis: ``st = [...,:st_dim]``, ``at = [...,st_dim:]`` (joint position targets).
    ``st`` layout: base_lin_vel(3), base_ang_vel(3), gravity_proj(3),
    joint_pos(nj), joint_vel(nj), joint_torque(na), with ``st_dim == 9 + 2*nj + na``.
    """
    return_numpy = isinstance(state_action_pair, np.ndarray)
    if return_numpy:
        x = torch.from_numpy(np.asarray(state_action_pair, dtype=np.float64))
    else:
        if torch.is_tensor(state_action_pair):
            x = state_action_pair
        else:
            x = torch.as_tensor(state_action_pair)
        if not torch.is_floating_point(x):
            x = x.float()

    device = x.device
    dtype = x.dtype

    if x.shape[-1] <= st_dim:
        raise ValueError(
            f"Last dim must be > st_dim ({st_dim}); got {x.shape[-1]}"
        )

    st = x[..., :st_dim]
    target_actions = x[..., st_dim:]
    nj = int(target_actions.shape[-1])
    na = st_dim - 9 - 2 * nj
    if na < 0 or st_dim != 9 + 2 * nj + na:
        raise ValueError(
            f"Bad layout: st_dim={st_dim}, nj={nj} -> need st_dim==9+2*nj+na"
        )

    def _bounds_1d(b):
        if torch.is_tensor(b):
            t = b.to(device=device, dtype=dtype).reshape(-1)
        else:
            t = torch.as_tensor(b, device=device, dtype=dtype).reshape(-1)
        return t

    jlo = _bounds_1d(joint_pos_min)
    jhi = _bounds_1d(joint_pos_max)
    tlo = _bounds_1d(tau_min)
    thi = _bounds_1d(tau_max)
    if jlo.shape != (nj,) or jhi.shape != (nj,):
        raise ValueError("joint_pos_min/max must have shape (nj,) matching action dim")
    if tlo.shape != (na,) or thi.shape != (na,):
        raise ValueError("tau_min/max must have shape (na,) matching actuator torques in st")

    def _normalize_to_minus_one_one_t(z, lo_1d, hi_1d):
        view_shape = (1,) * (z.ndim - 1) + (-1,)
        lo_b = lo_1d.view(view_shape)
        hi_b = hi_1d.view(view_shape)
        span = hi_b - lo_b
        valid = span > 1e-8
        out = torch.where(
            valid, 2.0 * (z - lo_b) / span - 1.0, torch.zeros_like(z, dtype=dtype, device=device)
        )
        return torch.clamp(out, -1.0, 1.0)

    def _normalize_symmetric_t(z, bound: float):
        if bound <= 0:
            raise ValueError("bound must be positive")
        return torch.clamp(z / bound, -1.0, 1.0)

    base_lin = st[..., 0:3]
    base_ang = st[..., 3:6]
    grav = st[..., 6:9]
    jq = st[..., 9 : 9 + nj]
    jv = st[..., 9 + nj : 9 + 2 * nj]
    tau = st[..., 9 + 2 * nj :]

    n_base_lin = _normalize_symmetric_t(base_lin, base_lin_vel_limit)
    n_base_ang = _normalize_symmetric_t(base_ang, base_ang_vel_limit)
    n_grav = _normalize_symmetric_t(grav, bound=1)
    n_jq = _normalize_to_minus_one_one_t(jq, jlo, jhi)
    n_jv = _normalize_symmetric_t(jv, joint_vel_limit)
    n_tau = _normalize_to_minus_one_one_t(tau, tlo, thi)
    st_out = torch.cat(
        [n_base_lin, n_base_ang, n_grav, n_jq, n_jv, n_tau], dim=-1
    )
    tgt_out = _normalize_to_minus_one_one_t(target_actions, jlo, jhi)

    if return_numpy:
        return st_out.detach().cpu().numpy(), tgt_out.detach().cpu().numpy()
    return st_out, tgt_out





    

def CreateGruWMInstance(config):
    world_model = RandomWorldStepGru(
        batch_size=config['world_model_training_params']['batch_size'],
        state_dim=config['robot_params']['state_dims'],
        contact_dim=config['robot_params']['contact_dims'],
        action_dim=config['robot_params']['action_dims'],
        embed_dim=config['world_model_arch_params']['embed_dim'],
        hidden_dim=config['world_model_arch_params']['gru_hidden_dim'],
        num_gru_layers=config['world_model_arch_params']['num_gru_layers'],
        mlp_dim=config['world_model_arch_params']['mlp_head_dim'],
        lr=config['world_model_training_params']['learning_rate'],
        weight_decay=config['world_model_training_params']['weight_decay'],
        device=config['device'],
        std_range=config['world_model_arch_params'].get('std_range', (0.03, 5)),
        std_init=config['world_model_arch_params'].get('std_init', 0.4)
    )
    return world_model
