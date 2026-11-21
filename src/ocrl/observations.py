"""Shared checkpoint-compatible observation and reproducibility helpers."""
from pathlib import Path
import hashlib
import json
import torch
from tensordict import TensorDict


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def load_config(path=Path('configs/default.json'), device='cpu'):
    config = json.loads(Path(path).read_text())
    config['device'] = device
    return config

def robot_state(env):
    data = env.unwrapped.scene['robot'].data
    return torch.cat([data.root_link_lin_vel_w, data.root_link_ang_vel_w,
                      data.projected_gravity_b, data.joint_pos, data.joint_vel,
                      data.actuator_force], dim=-1)

def actor_observation(obs):
    for key in ('policy', 'actor'):
        if key in obs.keys():
            return obs[key]
    raise KeyError('Expected actor or policy observation group')

class ObservationAdapter:
    """Preserve historical real-simulator input: normalized state + scaled last action."""
    def __init__(self, config, device):
        self.mean = torch.tensor(config['mean_state_action'][:96], device=device)
        self.std = torch.tensor(config['std_state_action'][:96], device=device)
        self.scale = torch.tensor(config['scale'], device=device)
        self.offset = torch.tensor(config['offset'], device=device)

    def normalize(self, state):
        return (state - self.mean) / (self.std + 1e-5)

    def scale_action(self, action):
        return action * self.scale + self.offset

    def policy_obs(self, state, previous_scaled_action, command):
        features = torch.cat([self.normalize(state)[..., :67], previous_scaled_action, command], -1)
        if features.shape[-1] != 99:
            raise ValueError(f'Expected 99 policy features, got {features.shape[-1]}')
        return TensorDict({'policy': features}, batch_size=[features.shape[0]], device=features.device)


def portable_arguments(arguments):
    """Serialize paths relative to the invocation directory, not a user's machine."""
    import os
    return {key: os.path.relpath(value) if isinstance(value, Path) else value
            for key, value in vars(arguments).items()}
