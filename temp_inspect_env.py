import sys
sys.path.append("/Users/vyvaswath/Documents/Personal/Spring_2026/OCRL/Project/OCRL_Random_Agent")

import json
from mjlab.envs import ManagerBasedRlEnv
from data.baseline_policies.unitree_rl_mjlab.src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

cfg = make_velocity_env_cfg()
env = ManagerBasedRlEnv(cfg, device="cpu")

print("num_envs:", env.num_envs)
#print("observation_space:", env.observation_space)
#print("action_space:", env.action_space)

obs, extras = env.reset()
print("reset() returned obs keys:", obs.keys())
print("reset() policy obs shape:", obs["actor"].shape)
print("reset() critic obs shape:", obs["critic"].shape)

import torch
actions = torch.zeros((env.num_envs, 29), device="cpu")
obs, rewards, dones, infos = env.step(actions)
print("step() returned obs keys:", obs.keys())
print("rewards shape:", rewards.shape)
print("dones shape:", dones.shape)
