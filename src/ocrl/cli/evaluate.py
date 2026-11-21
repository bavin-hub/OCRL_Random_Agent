"""Evaluate a historical G1 policy, record video, and export world-model predictions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ocrl.observations import portable_arguments, ObservationAdapter, actor_observation, load_config, robot_state, sha256


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('artifacts/historical/saved_models/model_3710.pt'))
    p.add_argument('--world-model', type=Path, default=Path('artifacts/historical/logs/ckpts/wm_gru_2026-05-09_23:10:44/wm-itr-ckpt-epoch_3710.pth'))
    p.add_argument('--config', type=Path, default=Path('configs/default.json'))
    p.add_argument('--output', type=Path, default=Path('runs/eval-3710'))
    p.add_argument('--task-id', default='Mjlab-Velocity-Flat-Unitree-G1')
    p.add_argument('--num-envs', type=int, default=1)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--viewer', choices=['none','native','viser'], default='none')
    p.add_argument('--video', action='store_true')
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--action-history', choices=['scaled','normalized'], default='scaled', help='Historical real rollout used scaled targets; imagination used normalized targets')
    p.add_argument('--prediction-horizon', type=int, help='Override forecast steps; warmup remains 32')
    p.add_argument('--command', type=float, nargs=3, metavar=('VX','VY','WZ'), help='Fixed body-frame velocity command')
    return p.parse_args()

class Policy:
    def __init__(self, env, actor, adapter, action_history="scaled", config=None):
        self.env, self.actor, self.adapter = env, actor, adapter
        self.previous = torch.zeros(env.num_envs, 29, device=env.device)
        self.action_history = action_history
        self.action_mean = torch.tensor(config['mean_state_action'][96:],device=env.device)
        self.action_std = torch.tensor(config['std_state_action'][96:],device=env.device)
    def __call__(self, obs):
        # The viewer and headless loop use exactly the same preprocessing.
        self.previous[self.env.unwrapped.episode_length_buf == 0] = 0
        command = actor_observation(obs)[:, -3:]
        features = self.adapter.policy_obs(robot_state(self.env), self.previous, command)
        with torch.inference_mode():
            action = self.actor(features, stochastic_output=False)
        self.previous = self.adapter.scale_action(action).clone()
        if self.action_history == 'normalized':
            self.previous = (self.previous-self.action_mean)/(self.action_std+1e-5)
        return action

def export_predictions(model, config, rows, output):
    """Use one uninterrupted trajectory; save normalized ground truth and deterministic forecasts."""
    warmup = config['world_model_training_params']['M']
    horizon = config['world_model_training_params']['N_pred']
    if len(rows) < warmup + horizon:
        return {'available': False, 'reason': 'No uninterrupted trajectory long enough for forecast'}
    x = torch.stack(rows[-(warmup+horizon):]).unsqueeze(0)
    ht = torch.zeros(config['world_model_arch_params']['num_gru_layers'], 1,
                     config['world_model_arch_params']['gru_hidden_dim'], device=x.device)
    predictions = []
    with torch.inference_mode():
        for t in range(warmup-1):
            ht = model(x[:,t:t+1], ht, predict=False)
        state = x[:,warmup-1:warmup,:96]
        for t in range(warmup-1, warmup+horizon-1):
            state, ht = model(torch.cat([state, x[:,t:t+1,96:]], -1), ht,
                              predict=True, sample=True, x_prev=state)
            predictions.append(state[0,0].cpu().numpy())
    pred = np.stack(predictions)
    truth = x[0,warmup:,:96].cpu().numpy()
    np.savez(output/'predictions.npz', predicted=pred, actual=truth,
             dt=config['world_model_training_params']['step_time_seconds'])
    return {'available': True, 'horizon_steps': horizon,
            'normalized_rmse': float(np.sqrt(np.mean((pred-truth)**2)))}

def main():
    args = parse_args()
    if min(args.steps, args.num_envs, args.width, args.height) < 1:
        raise ValueError('steps and num-envs must be positive')
    if args.video and (args.viewer != 'none' or args.num_envs != 1):
        raise ValueError('Video requires --viewer none --num-envs 1')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Check nvidia-smi and execution permissions.')
    if args.output.exists():
        raise FileExistsError(f'Choose a new --output directory: {args.output}')
    import mjlab.tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg
    from rsl_rl.models import MLPModel
    from ocrl.models.factory import CreateWorlModelInstance
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = load_config(args.config, args.device)
    if args.prediction_horizon is not None:
        if args.prediction_horizon < 1: raise ValueError('prediction-horizon must be positive')
        config['world_model_training_params']['N_pred'] = args.prediction_horizon
    cfg = load_env_cfg(args.task_id, play=False)
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.nconmax = 128
    cfg.viewer.width = args.width
    cfg.viewer.height = args.height
    cfg.observations['actor'].enable_corruption = False
    cfg.events.pop('push_robot', None)
    cfg.curriculum = {}
    if args.command:
        cmd = cfg.commands['twist']
        cmd.ranges.lin_vel_x = (args.command[0],)*2
        cmd.ranges.lin_vel_y = (args.command[1],)*2
        cmd.ranges.ang_vel_z = (args.command[2],)*2
        cmd.rel_standing_envs = 0.0
        cmd.rel_heading_envs = 0.0
        cmd.heading_command = False
        cmd.ranges.heading = None
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=args.device,
                         render_mode='rgb_array' if args.video else None), clip_actions=100.0)
    writer = None
    try:
        adapter = ObservationAdapter(config, args.device)
        obs = env.get_observations()
        sample = adapter.policy_obs(robot_state(env), torch.zeros(env.num_envs,29,device=args.device), actor_observation(obs)[:,-3:])
        actor = MLPModel(sample, {'actor':['policy']}, 'actor', 29, hidden_dims=[256]*3,
                         activation='elu', obs_normalization=True,
                         distribution_cfg={'class_name':'GaussianDistribution','init_std':1.0,'std_type':'scalar'}).to(args.device)
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        actor.load_state_dict(checkpoint['actor_state_dict'], strict=True)
        actor.eval()
        wm = CreateWorlModelInstance(config)
        checkpoint_wm = torch.load(args.world_model, map_location=args.device, weights_only=True)
        wm.load_state_dict(checkpoint_wm.get('model_state_dict', checkpoint_wm), strict=True)
        wm.eval()
        policy = Policy(env, actor, adapter, args.action_history, config)
        args.output.mkdir(parents=True)
        import importlib.metadata as package_metadata
        metadata = {'versions': {name:package_metadata.version(name) for name in ['torch','mjlab','rsl-rl-lib','mujoco','mujoco-warp']}, 'arguments':portable_arguments(args), 'policy_sha256':sha256(args.checkpoint),
                    'world_model_sha256':sha256(args.world_model), 'config':config,
                    'evaluation_protocol':'No observation corruption or pushes; simulator reward; fixed seed',
                    'policy_iteration':checkpoint.get('iter')}
        (args.output/'manifest.json').write_text(json.dumps(metadata,default=str,indent=2))
        if args.viewer != 'none':
            from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
            (NativeMujocoViewer if args.viewer=='native' else ViserPlayViewer)(env, policy).run()
            return
        if args.video:
            import imageio.v2 as imageio
            writer = imageio.get_writer(args.output/'rollout.mp4', fps=round(1/env.unwrapped.step_dt), codec='libx264')
        rows, best_rows, trace, episodes = [], [], [], []
        returns = torch.zeros(env.num_envs,device=args.device)
        lengths = torch.zeros(env.num_envs,device=args.device,dtype=torch.long)
        act_mean = torch.tensor(config['mean_state_action'][96:], device=args.device)
        act_std = torch.tensor(config['std_state_action'][96:], device=args.device)
        limit = config['world_model_training_params']['M'] + config['world_model_training_params']['N_pred']
        with torch.inference_mode():
            for step in range(args.steps):
                state = robot_state(env).clone()
                command = actor_observation(obs)[:,-3:].clone()
                body_velocity = env.unwrapped.scene['robot'].data.root_link_lin_vel_b.clone()
                body_angular = env.unwrapped.scene['robot'].data.root_link_ang_vel_b.clone()
                actions = policy(obs)
                normalized_action = (adapter.scale_action(actions)-act_mean)/(act_std+1e-5)
                rows.append(torch.cat([adapter.normalize(state)[0],normalized_action[0]]).clone())
                rows = rows[-limit:]
                if len(rows) >= len(best_rows): best_rows = list(rows)
                obs, rewards, dones, extras = env.step(actions)
                returns += rewards
                lengths += 1
                trace.append([step, rewards.mean().item(),
                              *((body_velocity[:,:2]-command[:,:2]).square().mean(0).sqrt().tolist()),
                              (body_angular[:,2]-command[:,2]).square().mean().sqrt().item()])
                for i in dones.nonzero().flatten().tolist():
                    timeout = bool(extras.get('time_outs',torch.zeros_like(dones))[i])
                    episodes.append({'env':i,'return':returns[i].item(),'length':lengths[i].item(),'timeout':timeout})
                    returns[i]=0; lengths[i]=0
                if dones[0]: rows=[]
                if writer:
                    frame = env.unwrapped.render()
                    writer.append_data(frame)
                    if step == 0: imageio.imwrite(args.output/'poster.png',frame)
        np.savetxt(args.output/'metrics.csv',np.asarray(trace),delimiter=',',
                   header='step,simulator_reward,vx_rmse,vy_rmse,yaw_rate_rmse',comments='')
        summary = {'completed_episodes':episodes, 'unfinished_episode_lengths':lengths.tolist(),
                   'mean_step_reward':float(np.mean(np.asarray(trace)[:,1])),
                   'world_model':export_predictions(wm,config,best_rows,args.output)}
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary,indent=2))
    finally:
        if writer: writer.close()
        env.close()

if __name__ == '__main__':
    main()
