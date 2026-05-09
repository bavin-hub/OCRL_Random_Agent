# contains plot, save, load utils 
import os, json
from pathlib import Path
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

    ckpts_dir = os.path.join(os.path.join(os.getcwd(), f"logs/ckpts/{model_dir_name}"))
    if not os.path.isdir(ckpts_dir):
        print("Creating ckpts dir")
        os.makedirs(ckpts_dir)

    return models_dir, plots_dir, ckpts_dir


def scale_policy_actions(actions, scale, offset):
    return (actions*scale) + offset


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


def load_ppo_policy_from_checkpoint(
    algo,
    checkpoint_path,
    *,
    load_optimizer: bool = True,
    strict: bool = True,
):

    p = Path(checkpoint_path)
    if not p.is_file():
        p = Path(os.getcwd()) / checkpoint_path
    if not p.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {checkpoint_path}")

    loaded = torch.load(str(p), map_location="cpu", weights_only=False)
    load_cfg = {
        "actor": True,
        "critic": True,
        "optimizer": load_optimizer,
        "iteration": False,
        "rnd": True,
    }
    algo.load(loaded, load_cfg, strict=strict)
    return loaded.get("iter")


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

    def plot_joint_dict_figure(true_dict, pred_dict, title: str, fname_suffix: str):
        fig_j, axes_j = plt.subplots(
            len(JOINT_PICK), 1, figsize=(20, 5 * len(JOINT_PICK)), sharex=True
        )
        if len(JOINT_PICK) == 1:
            axes_j = [axes_j]
        for ax, j in zip(axes_j, JOINT_PICK):
            ax.plot(time_steps, true_dict[j], label="true", color=c_true, linewidth=lw)
            ax.plot(time_steps, pred_dict[j], label="pred", linestyle=":", color=c_pred, linewidth=lw)
            ax.set_ylabel(f"joint {j}")
            mark_history(ax)
        axes_j[-1].set_xlabel("time step (1-based)")
        fig_j.suptitle(title)
        fig_j.tight_layout()
        save_plot_figure(fig_j, model_dir_name, f"{model_name}_{fname_suffix}")
        plt.show()

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

    if proj_grav_true is not None and proj_grav_pred is not None:
        gx_t, gy_t, gz_t = proj_grav_true
        gx_p, gy_p, gz_p = proj_grav_pred
        fig_g, axes_g = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
        grav_true = [gx_t, gy_t, gz_t]
        grav_pred = [gx_p, gy_p, gz_p]
        labels_g = ["proj_gravity_x", "proj_gravity_y", "proj_gravity_z"]
        for k, ax in enumerate(axes_g):
            ax.plot(time_steps, grav_true[k], label="true", color=c_true, linewidth=lw)
            ax.plot(time_steps, grav_pred[k], label="pred", linestyle=":", color=c_pred, linewidth=lw)
            ax.set_ylabel(labels_g[k])
            mark_history(ax)
        axes_g[-1].set_xlabel("time step (1-based)")
        fig_g.suptitle("Projected gravity (body frame)")
        fig_g.tight_layout()
        save_plot_figure(fig_g, model_dir_name, f"{model_name}_proj_gravity")
        plt.show()

    if true_joint_vel is not None and pred_joint_vel is not None:
        plot_joint_dict_figure(
            true_joint_vel, pred_joint_vel, "Joint velocities (subset)", "joint_vel"
        )

    if true_joint_tau is not None and pred_joint_tau is not None:
        plot_joint_dict_figure(
            true_joint_tau, pred_joint_tau, "Joint torques (subset)", "joint_torques"
        )


# def z_norm(state_action_pair, mean, std):
#     # print(type(state_action_pair))
#     state_action_pair = (state_action_pair - mean) / (std + 1e-8)
#     return state_action_pair.astype(np.float32)


def z_norm(state, action, state_mean, state_std, action_mean, action_std):
    # print(type(state_action_pair))
    state_norm = (state - state_mean) / (state_std + 1e-5)
    action_norm = (action - action_mean) / (action_std + 1e-5)
    return state_norm, action_norm


def denormalize_z_norm(state_norm, action_norm, state_mean, state_std, action_mean, action_std):
    """Inverse of ``z_norm``: recover state and action from z-normalized tensors.

    Matches ``denormalize`` conventions: NumPy in → NumPy out on CPU; torch tensors keep
    device/dtype (promoting ``action_norm`` to match ``state_norm``). If a tensor has rank 3
    and ``shape[1] == 1``, the singleton middle axis is squeezed (same as ``denormalize`` /
    ``last_action`` handling).
    """
    eps = 1e-5
    return_numpy = isinstance(state_norm, np.ndarray)
    if return_numpy:
        if not isinstance(action_norm, np.ndarray):
            raise TypeError(
                "denormalize_z_norm: state_norm is ndarray but action_norm is not; "
                "use matching types."
            )
        s_n = torch.from_numpy(np.asarray(state_norm, dtype=np.float64))
        a_n = torch.from_numpy(np.asarray(action_norm, dtype=np.float64))
    else:
        if torch.is_tensor(state_norm):
            s_n = state_norm
        else:
            s_n = torch.as_tensor(state_norm)
        if not torch.is_floating_point(s_n):
            s_n = s_n.float()
        if torch.is_tensor(action_norm):
            a_n = action_norm
        else:
            a_n = torch.as_tensor(action_norm)
        if not torch.is_floating_point(a_n):
            a_n = a_n.float()

    device = s_n.device
    dtype = s_n.dtype
    a_n = a_n.to(device=device, dtype=dtype)

    if s_n.ndim == 3 and s_n.shape[1] == 1:
        s_n = s_n.squeeze(1)
    if a_n.ndim == 3 and a_n.shape[1] == 1:
        a_n = a_n.squeeze(1)

    if s_n.shape[:-1] != a_n.shape[:-1]:
        raise ValueError(
            f"state_norm leading shape {tuple(s_n.shape[:-1])} must match "
            f"action_norm {tuple(a_n.shape[:-1])}"
        )

    def _mean_std_1d(mean, std, last_dim: int, *, label: str):
        if torch.is_tensor(mean):
            m = mean.to(device=device, dtype=dtype).reshape(-1)
        else:
            m = torch.as_tensor(mean, device=device, dtype=dtype).reshape(-1)
        if torch.is_tensor(std):
            s = std.to(device=device, dtype=dtype).reshape(-1)
        else:
            s = torch.as_tensor(std, device=device, dtype=dtype).reshape(-1)
        if m.shape[0] != last_dim:
            raise ValueError(f"{label}_mean length {m.shape[0]} != last dim {last_dim}")
        if s.shape[0] != last_dim:
            raise ValueError(f"{label}_std length {s.shape[0]} != last dim {last_dim}")
        return m, s

    sm, ss = _mean_std_1d(state_mean, state_std, s_n.shape[-1], label="state")
    am, as_ = _mean_std_1d(action_mean, action_std, a_n.shape[-1], label="action")

    view_m = (1,) * (s_n.ndim - 1) + (-1,)
    view_a = (1,) * (a_n.ndim - 1) + (-1,)
    state = s_n * (ss.view(view_m) + eps) + sm.view(view_m)
    action = a_n * (as_.view(view_a) + eps) + am.view(view_a)

    if return_numpy:
        return state.detach().cpu().numpy(), action.detach().cpu().numpy()
    return state, action


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
    """Normalize a concatenated [st | at] vector like `normalize_st_and_target_actions` in base.py.

    Last axis: ``st = [...,:st_dim]``, ``at = [...,st_dim:]`` (joint position targets).
    ``st`` layout matches base: base_lin_vel(3), base_ang_vel(3), gravity_proj(3),
    joint_pos(nj), joint_vel(nj), joint_torque(na), with ``st_dim == 9 + 2*nj + na``.

    Default path: ``state_action_pair`` is a ``torch.Tensor`` (e.g. on CUDA); bounds may be
    tensors or array-likes and are moved to the same device and dtype as ``state_action_pair``.
    NumPy array input is still supported and returns NumPy outputs on CPU.
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


def denormalize(
    st_pred,
    joint_pos_min,
    joint_pos_max,
    tau_min,
    tau_max,
    *,
    last_action=None,
    st_dim: int = 96,
    base_lin_vel_limit: float = 4.0,
    base_ang_vel_limit: float = 10.0,
    joint_vel_limit: float = 15.0,
):
    # st_pred -> (num_envs, 1, state_dims) or (num_envs, state_dims)
    # last_action: optional normalized joint targets (same as tgt_out in minmax_norm_state_action_pair),
    #   shape (..., nj) or (..., 1, nj); denormalized with joint_pos_min/max to raw joint commands.
    # Returns st_out only, or (st_out, at_out) if last_action is not None (same numpy/torch dtype rules).
    return_numpy = isinstance(st_pred, np.ndarray)
    if return_numpy:
        n = torch.from_numpy(np.asarray(st_pred, dtype=np.float64))
    else:
        if torch.is_tensor(st_pred):
            n = st_pred
        else:
            n = torch.as_tensor(st_pred)
        if not torch.is_floating_point(n):
            n = n.float()

    device = n.device
    dtype = n.dtype

    if n.ndim == 3 and n.shape[1] == 1:
        n = n.squeeze(1)

    if n.shape[-1] != st_dim:
        raise ValueError(f"expected last dim st_dim={st_dim}; got {n.shape[-1]}")

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
    nj = int(jlo.shape[0])
    na = int(tlo.shape[0])
    if jhi.shape[0] != nj:
        raise ValueError("joint_pos_max must match joint_pos_min length")
    if thi.shape[0] != na:
        raise ValueError("tau_max must match tau_min length")
    if st_dim != 9 + 2 * nj + na:
        raise ValueError(
            f"Bad layout: st_dim={st_dim}, nj={nj}, na={na} -> need st_dim==9+2*nj+na"
        )

    na_t = None
    if last_action is not None:
        if return_numpy:
            na_t = torch.from_numpy(np.asarray(last_action, dtype=np.float64))
        else:
            if torch.is_tensor(last_action):
                na_t = last_action
            else:
                na_t = torch.as_tensor(last_action)
            if not torch.is_floating_point(na_t):
                na_t = na_t.float()
        na_t = na_t.to(device=device, dtype=dtype)
        if na_t.ndim == 3 and na_t.shape[1] == 1:
            na_t = na_t.squeeze(1)
        if na_t.shape[-1] != nj:
            raise ValueError(f"last_action last dim must be nj={nj}; got {na_t.shape[-1]}")
        if na_t.shape[:-1] != n.shape[:-1]:
            raise ValueError(
                f"last_action leading shape {tuple(na_t.shape[:-1])} must match st_pred {tuple(n.shape[:-1])}"
            )

    def _denorm_symmetric_t(norm_z, bound):
        b = torch.as_tensor(bound, device=device, dtype=dtype)
        return norm_z * b

    def _denorm_from_minus_one_one_t(norm_z, lo_1d, hi_1d):
        view_shape = (1,) * (norm_z.ndim - 1) + (-1,)
        lo_b = lo_1d.view(view_shape)
        hi_b = hi_1d.view(view_shape)
        span = hi_b - lo_b
        valid = span > 1e-8
        return torch.where(
            valid,
            (norm_z + 1.0) * 0.5 * span + lo_b,
            lo_b,
        )

    n_base_lin = n[..., 0:3]
    n_base_ang = n[..., 3:6]
    n_grav = n[..., 6:9]
    n_jq = n[..., 9 : 9 + nj]
    n_jv = n[..., 9 + nj : 9 + 2 * nj]
    n_tau = n[..., 9 + 2 * nj :]

    base_lin = _denorm_symmetric_t(n_base_lin, base_lin_vel_limit)
    base_ang = _denorm_symmetric_t(n_base_ang, base_ang_vel_limit)
    # matches minmax_norm_state_action_pair: gravity uses bound=1, not gravity_limit
    grav = _denorm_symmetric_t(n_grav, 1.0)
    jq = _denorm_from_minus_one_one_t(n_jq, jlo, jhi)
    jv = _denorm_symmetric_t(n_jv, joint_vel_limit)
    tau = _denorm_from_minus_one_one_t(n_tau, tlo, thi)
    st_out = torch.cat(
        [base_lin, base_ang, grav, jq, jv, tau], dim=-1
    )

    if na_t is not None:
        at_out = _denorm_from_minus_one_one_t(na_t, jlo, jhi)
        if return_numpy:
            return st_out.detach().cpu().numpy(), at_out.detach().cpu().numpy()
        return st_out, at_out

    if return_numpy:
        return st_out.detach().cpu().numpy()
    return st_out





    