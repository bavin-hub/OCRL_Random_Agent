"""State and action normalization."""
import numpy as np
import torch

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
