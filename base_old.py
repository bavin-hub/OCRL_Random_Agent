"""Base class for environment viewers."""

from __future__ import annotations

import contextlib
import time
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Any, Optional, Protocol

import torch

import numpy as np
import mujoco
import os, sqlite3, json


if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg


class VerbosityLevel(IntEnum):
  SILENT = 0
  INFO = 1
  DEBUG = 2


class Timer:
  def __init__(self):
    self._previous_time = time.time()
    self._measured_time = 0.0

  def tick(self):
    curr_time = time.time()
    self._measured_time = curr_time - self._previous_time
    self._previous_time = curr_time
    return self._measured_time

  @contextlib.contextmanager
  def measure_time(self):
    start_time = time.time()
    yield
    self._measured_time = time.time() - start_time

  @property
  def measured_time(self):
    return self._measured_time


class EnvProtocol(Protocol):
  """Interface we expect from RL environments, which can be either vanilla
  `ManagerBasedRlEnv` objects or wrapped with `VideoRecorder`,
  `RslRlVecEnvWrapper`, etc."""

  num_envs: int

  @property
  def device(self) -> torch.device | str: ...

  @property
  def cfg(self) -> ManagerBasedRlEnvCfg: ...

  @property
  def unwrapped(self) -> Any: ...

  def get_observations(self) -> Any: ...
  def step(self, actions: torch.Tensor) -> tuple[Any, ...]: ...
  def reset(self) -> Any: ...
  def close(self) -> None: ...


class PolicyProtocol(Protocol):
  def __call__(self, obs: torch.Tensor) -> torch.Tensor: ...


class ViewerAction(Enum):
  RESET = "reset"
  TOGGLE_PAUSE = "toggle_pause"
  SPEED_UP = "speed_up"
  SPEED_DOWN = "speed_down"
  PREV_ENV = "prev_env"
  NEXT_ENV = "next_env"
  CUSTOM = "custom"


class BaseViewer(ABC):
  """Abstract base class for environment viewers."""

  SPEED_MULTIPLIERS = [0.01, 0.016, 0.025, 0.04, 0.063, 0.1, 0.16, 0.25, 0.4, 0.63, 1.0]

  def __init__(
    self,
    env: EnvProtocol,
    policy: PolicyProtocol,
    frame_rate: float = 30.0,
    verbosity: int = VerbosityLevel.SILENT,
  ):
    self.env = env
    self.policy = policy
    self.frame_rate = frame_rate
    self.frame_time = 1.0 / frame_rate
    self.verbosity = VerbosityLevel(verbosity)
    self.cfg = env.cfg.viewer

    # Loop state.
    self._is_paused = False
    self._step_count = 0

    # Timing.
    self._timer = Timer()
    self._sim_timer = Timer()
    self._render_timer = Timer()
    self._time_until_next_frame = 0.0

    self._speed_index = self.SPEED_MULTIPLIERS.index(1.0)
    self._time_multiplier = self.SPEED_MULTIPLIERS[self._speed_index]

    # Perf tracking.
    self._frame_count = 0
    self._last_fps_log_time = 0.0
    self._accumulated_sim_time = 0.0
    self._accumulated_render_time = 0.0

    # FPS tracking.
    self._smoothed_fps: float = 0.0
    self._fps_accum_frames: int = 0
    self._fps_accum_time: float = 0.0
    self._fps_last_frame_time: Optional[float] = None
    self._fps_update_interval: float = 0.5
    self._fps_alpha: float = 0.35

    # Thread-safe action queue (drained in main loop).
    self._actions: deque[tuple[ViewerAction, Optional[Any]]] = deque()

    # Rollout saving: one SQLite file per trajectory under
    # `{db_dir}/{buffer_size_d * transitions_per_trajectory}_transitions/`.
    self.db_path = None
    self._rollout_output_dir: Optional[str] = None
    self.trajectory_ctr = 0
    self.local_buffer = []
    self.incremental_step = 0
    self.trajectory = []
    self.buffer_size_d = 1000
    self.transitions_per_trajectory = 1000
    self.idx = 0
    self._replay_initial_state_set = False

  # Abstract hooks every concrete viewer must implement.

  @abstractmethod
  def setup(self) -> None: ...
  @abstractmethod
  def sync_env_to_viewer(self) -> None: ...
  @abstractmethod
  def sync_viewer_to_env(self) -> None: ...
  @abstractmethod
  def close(self) -> None: ...
  @abstractmethod
  def is_running(self) -> bool: ...

  # Logging.

  def log(self, message: str, level: VerbosityLevel = VerbosityLevel.INFO) -> None:
    if self.verbosity >= level:
      print(message)

  # Public controls.

  def request_reset(self) -> None:
    self._actions.append((ViewerAction.RESET, None))

  def request_toggle_pause(self) -> None:
    self._actions.append((ViewerAction.TOGGLE_PAUSE, None))

  def request_speed_up(self) -> None:
    self._actions.append((ViewerAction.SPEED_UP, None))

  def request_speed_down(self) -> None:
    self._actions.append((ViewerAction.SPEED_DOWN, None))

  def request_action(self, name: str, payload: Optional[Any] = None) -> None:
    """Viewer-specific actions (e.g., PREV_ENV/NEXT_ENV for native)."""
    try:
      action = ViewerAction[name]
    except KeyError:
      action = ViewerAction.CUSTOM
    self._actions.append((action, payload))


  def save_trajectory(self) -> None:
    """Write the current `local_buffer` to its own SQLite file (one trajectory per DB)."""
    if self._rollout_output_dir is None:
      return
    out_path = os.path.join(
      self._rollout_output_dir, f"trajectory_{self.trajectory_ctr:05d}.db"
    )
    if os.path.isfile(out_path):
      os.remove(out_path)
    conn = sqlite3.connect(out_path)
    try:
      cursor = conn.cursor()
      cursor.execute("CREATE TABLE PretrainingData (trajectories TEXT)")
      # One trajectory per file: always use trajectory-1 so loaders that key by
      # row index (e.g. trajectory-{file_k+1} with a single row per DB) still work.
      single_roll_out = json.dumps({"trajectory-1": self.local_buffer})
      cursor.execute(
        "INSERT INTO PretrainingData (trajectories) VALUES (?)", (single_roll_out,)
      )
      conn.commit()
    finally:
      conn.close()
    self.db_path = out_path


  def _set_replay_initial_state(self) -> None:
    """Set sim state from trajectory[0][0] (state at t=0) so replay matches recording."""
    if not self.trajectory or len(self.trajectory) < 1:
      return
    base_env = self.env.unwrapped
    robot = base_env.scene["robot"]
    device = base_env.device
    env_id = slice(0, 1)  # env 0
    first_st = np.array(self.trajectory[0][0], dtype=np.float64)
    # state layout: base_lin_vel(3), base_ang_vel(3), gravity_proj(3), joint_pos(nj), joint_vel(nj), joint_torque(na)
    nj = (len(first_st) - 9 - robot.num_actuators) // 2
    if nj <= 0:
      return
    base_lin_vel = first_st[0:3]
    base_ang_vel = first_st[3:6]
    joint_pos = first_st[9 : 9 + nj]
    joint_vel = first_st[9 + nj : 9 + 2 * nj]
    root_vel = torch.tensor(
      np.concatenate([base_lin_vel, base_ang_vel]),
      dtype=torch.float,
      device=device,
    ).unsqueeze(0)
    robot.write_root_link_velocity_to_sim(root_vel, env_ids=env_id)
    joint_pos_t = torch.tensor(joint_pos, dtype=torch.float, device=device).unsqueeze(0)
    joint_vel_t = torch.tensor(joint_vel, dtype=torch.float, device=device).unsqueeze(0)
    robot.write_joint_state_to_sim(joint_pos_t, joint_vel_t, env_ids=env_id)


  def load_trajectory(self, db_path: str, row_index: int = 0, trajectory_key: str = "trajectory-1"):
    """Load one trajectory from the DB. Returns list of (st, ct, at)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM PretrainingData")
    rows = cursor.fetchall()
    conn.close()
    if not rows:
      raise FileNotFoundError(f"No rows in DB at {db_path}")
    raw = json.loads(rows[row_index][0])
    if trajectory_key not in raw:
      raise KeyError(f"Key {trajectory_key!r} not in row; keys: {list(raw.keys())}")
    return raw[trajectory_key]


  def _normalize_to_minus_one_one(self, x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Map each element from [lo, hi] to [-1, 1]. Constant columns -> 0."""
    span = hi - lo
    out = np.zeros_like(x, dtype=np.float64)
    mask = span > 1e-8
    out[mask] = 2.0 * (x[mask] - lo[mask]) / span[mask] - 1.0
    return np.clip(out, -1.0, 1.0)

  def _normalize_symmetric(self, x: np.ndarray, bound: float) -> np.ndarray:
    """Map [-bound, bound] -> [-1, 1]."""
    if bound <= 0:
      raise ValueError("bound must be positive")
    return np.clip(x / bound, -1.0, 1.0)

    


  def normalize_st_and_target_actions(
    self,
    st: np.ndarray,
    target_actions: np.ndarray,
    joint_pos_min: np.ndarray,
    joint_pos_max: np.ndarray,
    tau_min: np.ndarray,
    tau_max: np.ndarray,
    *,
    base_lin_vel_limit: float = 6.0,
    base_ang_vel_limit: float = 12.0,
    gravity_limit: float = 9.81,
    joint_vel_limit: float = 20.0,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Scale `st` and `target_actions` to approximately [-1, 1].
    `st` layout: base_lin_vel(3), base_ang_vel(3), gravity_proj(3),
    joint_pos(nj), joint_vel(nj), joint_torque(na).
    Uses symmetric bounds for base velocities, gravity (per component), and joint_vel.
    Uses per-joint [min, max] for joint_pos (hard or soft — pass the arrays you want).
    Uses per-actuator [tau_min, tau_max] for torques (e.g. from mj_model.actuator_forcerange).
    `target_actions` are joint position targets: same joint_pos_min / joint_pos_max as joint_pos block.
    """
    st = np.asarray(st, dtype=np.float64).reshape(-1)
    target_actions = np.asarray(target_actions, dtype=np.float64).reshape(-1)
    nj = len(target_actions)
    na = len(st) - 9 - 2 * nj
    if na < 0 or len(st) != 9 + 2 * nj + na:
      raise ValueError(
        f"Bad shapes: len(st)={len(st)}, nj={nj} -> need len(st)==9+2*nj+na"
      )
    if joint_pos_min.shape != (nj,) or joint_pos_max.shape != (nj,):
      raise ValueError("joint_pos_min/max must have shape (nj,) matching target_actions")
    if tau_min.shape != (na,) or tau_max.shape != (na,):
      raise ValueError("tau_min/max must have shape (na,) matching actuator torques in st")
    base_lin = st[0:3]
    base_ang = st[3:6]
    grav = st[6:9]
    jq = st[9 : 9 + nj]
    jv = st[9 + nj : 9 + 2 * nj]
    tau = st[9 + 2 * nj :]
    n_base_lin = self._normalize_symmetric(base_lin, base_lin_vel_limit)
    n_base_ang = self._normalize_symmetric(base_ang, base_ang_vel_limit)
    n_grav = self._normalize_symmetric(grav, gravity_limit)
    n_jq = self._normalize_to_minus_one_one(jq, joint_pos_min, joint_pos_max)
    n_jv = self._normalize_symmetric(jv, joint_vel_limit)
    n_tau = self._normalize_to_minus_one_one(tau, tau_min, tau_max)
    st_out = np.concatenate([n_base_lin, n_base_ang, n_grav, n_jq, n_jv, n_tau], axis=-1)
    tgt_out = self._normalize_to_minus_one_one(target_actions, joint_pos_min, joint_pos_max)
    return st_out, tgt_out


  def extract_observation_vectors(self, env, st, policy_at, ct, env_id=0):
    """Extract [base_lin_vel, base_ang_vel, gravity_proj, joint_pos, joint_vel, joint_torque],
       [body_contact, foot_heights, foot_velocities], and target actions from current state.

    Call this after env.step(actions) so actuator forces and contacts are updated.
    Returns numpy arrays for the given env_id (for use in buffers/logging).
    """
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    mjm = base_env.sim.mj_model
    mjd = base_env.sim.mj_data

    # --- State vector: base_lin_vel, base_ang_vel, gravity_proj, joint_pos, joint_vel, joint_torque ---
    base_lin_vel = robot.data.root_link_lin_vel_w[env_id].cpu().numpy()      # (3,)
    base_ang_vel = robot.data.root_link_ang_vel_w[env_id].cpu().numpy()     # (3,)
    gravity_proj = robot.data.projected_gravity_b[env_id].cpu().numpy()      # (3,)
    joint_pos = robot.data.joint_pos[env_id].cpu().numpy()                   # (num_joints,)
    joint_vel = robot.data.joint_vel[env_id].cpu().numpy()                   # (num_joints,)
    joint_torque = robot.data.actuator_force[env_id].cpu().numpy()           # (num_actuators,) in actuation space

    state_vec = np.concatenate([
        base_lin_vel, base_ang_vel, gravity_proj,
        joint_pos, joint_vel, joint_torque,
    ], axis=-1)

    # --- Contact vector: body_contact, foot_heights, foot_velocities ---
    # Body contact flags from MuJoCo contact list (same logic as extract_and_print_state)
    contact_flags = np.zeros(mjm.nbody, dtype=np.int64)
    for i in range(mjd.ncon):
        con = mjd.contact[i]
        contact_flags[mjm.geom_bodyid[con.geom1]] = 1
        contact_flags[mjm.geom_bodyid[con.geom2]] = 1
    body_contact = contact_flags[:26]  # G1 robot body count

    # Foot heights and velocities from left_foot / right_foot sites
    try:
        left_ix = robot.site_names.index("left_foot")
        right_ix = robot.site_names.index("right_foot")
    except ValueError:
        left_ix, right_ix = 0, 1
    foot_site_ix = [left_ix, right_ix]
    foot_heights = robot.data.site_pos_w[env_id, foot_site_ix, 2].cpu().numpy()  # (2,)
    foot_velocities = np.linalg.norm(
        robot.data.site_lin_vel_w[env_id, foot_site_ix, :].cpu().numpy(), axis=-1
    )  # (2,)

    contact_vec = np.concatenate([body_contact, foot_heights, foot_velocities], axis=-1)

    # --- Target actions (joint position targets fed to the robot) ---
    target_actions = robot.data.joint_pos_target[env_id].cpu().numpy()

    # print(policy_at.cpu().numpy().squeeze().shape)

    # print('\n\nafter step : ', target_actions)


    # normalize state and target actions
    jmin = robot.data.joint_pos_limits[env_id].cpu().numpy()[:, 0]   # or soft_joint_pos_limits[..., 0]
    jmax = robot.data.joint_pos_limits[env_id].cpu().numpy()[:, 1]   # or soft_joint_pos_limits[..., 1]

    ctrl_ids = robot.data.indexing.ctrl_ids
    idx = ctrl_ids.cpu().numpy() if hasattr(ctrl_ids, "cpu") else np.asarray(ctrl_ids)
    tau_min = mjm.actuator_forcerange[idx, 0]
    tau_max = mjm.actuator_forcerange[idx, 1]

    st_n, target_n = self.normalize_st_and_target_actions(
    np.asarray(st, dtype=np.float64),
    target_actions,
    jmin,
    jmax,
    tau_min,
    tau_max,)

    self.local_buffer.append((st_n.tolist(), 
                              ct.tolist(), 
                              target_n.tolist(),
                              policy_at.cpu().numpy().squeeze().tolist()))

    # return state_vec, contact_vec, target_actions


  # Core loop.

  def get_state(self, env):
    env_id=0
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    mjm = base_env.sim.mj_model
    mjd = base_env.sim.mj_data

    base_lin_vel = robot.data.root_link_lin_vel_w[env_id].cpu().numpy()      # (3,)
    base_ang_vel = robot.data.root_link_ang_vel_w[env_id].cpu().numpy()     # (3,)
    gravity_proj = robot.data.projected_gravity_b[env_id].cpu().numpy()      # (3,)
    joint_pos = robot.data.joint_pos[env_id].cpu().numpy()                   # (num_joints,)
    joint_vel = robot.data.joint_vel[env_id].cpu().numpy()                   # (num_joints,)
    joint_torque = robot.data.actuator_force[env_id].cpu().numpy()           # (num_actuators,) in actuation space

    state_vec = np.concatenate([
        base_lin_vel, base_ang_vel, gravity_proj,
        joint_pos, joint_vel, joint_torque,
    ], axis=-1)

    return state_vec


  def get_contact(self, env):
    env_id=0
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    mjm = base_env.sim.mj_model
    mjd = base_env.sim.mj_data

    contact_flags = np.zeros(mjm.nbody, dtype=np.int64)
    for i in range(mjd.ncon):
        con = mjd.contact[i]
        contact_flags[mjm.geom_bodyid[con.geom1]] = 1
        contact_flags[mjm.geom_bodyid[con.geom2]] = 1
    body_contact = contact_flags[:26]  # G1 robot body count

    # Foot heights and velocities from left_foot / right_foot sites
    try:
        left_ix = robot.site_names.index("left_foot")
        right_ix = robot.site_names.index("right_foot")
    except ValueError:
        left_ix, right_ix = 0, 1
    foot_site_ix = [left_ix, right_ix]
    foot_heights = robot.data.site_pos_w[env_id, foot_site_ix, 2].cpu().numpy()  # (2,)
    foot_velocities = np.linalg.norm(
        robot.data.site_lin_vel_w[env_id, foot_site_ix, :].cpu().numpy(), axis=-1
    )  # (2,)

    contact_vec = np.concatenate([body_contact, foot_heights, foot_velocities], axis=-1)
    return contact_vec


  def step_simulation(self, replay=False) -> None:
    if self._is_paused:
      return
    with torch.no_grad():
      with self._sim_timer.measure_time():

        if replay:
          if len(self.trajectory) == 0:
            self.trajectory = self.load_trajectory(db_path="/home/bavin/cmu/sem2/ocrl/OCRL_Random_Agent/data/pretraining_rollouts/rollout_buffer.db")
            self._replay_initial_state_set = False
          if len(self.trajectory) == 0:
            self._step_count += 1
            return
          # Set initial state from first frame once (and again after reset when idx wraps)
          if not self._replay_initial_state_set:
            self._set_replay_initial_state()
            self._replay_initial_state_set = True
          # Bounds check: wrap to start when trajectory ends
          if self.idx >= len(self.trajectory):
            self.idx = 0
            self._set_replay_initial_state()
          # Feed the action for current step
          single_step_policy_at = torch.tensor([self.trajectory[self.idx][3]], device=self.env.unwrapped.device, dtype=torch.float)
          self.env.step(single_step_policy_at)
          self.idx += 1
        else:
          base_env = self.env.unwrapped

          if not getattr(self, "_printed_joint_actuator_limits", False):
            self._printed_joint_actuator_limits = True
            robot = base_env.scene["robot"]
            env_id = 0
            jpl = robot.data.joint_pos_limits[env_id].cpu().numpy()
            sjpl = robot.data.soft_joint_pos_limits[env_id].cpu().numpy()
            print("\n=== Joint position limits [min, max] per joint (rad) ===")
            print("hard:", jpl)
            print("soft:", sjpl)
            mjm = base_env.sim.mj_model
            print("\n=== Actuator forcerange [min, max] per ctrl (matches actuator order in mj_model) ===")
            print(mjm.actuator_forcerange)
            print("actuator_forcelimited:", mjm.actuator_forcelimited)
            # MuJoCo usually has no per-DOF velocity limits like jnt_range; use rollouts or chosen bounds.
            print("\n(No standard sim-wide joint velocity min/max in mj_model; check your MJCF if you added any.)\n")


          obs = self.env.get_observations()
          actions = self.policy(obs)

          # robot = base_env.scene["robot"]
          # mjm = base_env.sim.mj_model
          # mjd = base_env.sim.mj_data
          # env_id = 0

          # base_lin_vel = robot.data.root_link_lin_vel_w[env_id].cpu().numpy()
          # base_ang_vel = robot.data.root_link_ang_vel_w[env_id].cpu().numpy()
          # gravity_proj = robot.data.projected_gravity_b[env_id].cpu().numpy()
          # joint_pos = robot.data.joint_pos[env_id].cpu().numpy()
          # joint_vel = robot.data.joint_vel[env_id].cpu().numpy()
          # joint_torque = robot.data.actuator_force[env_id].cpu().numpy()

          # state_vec = np.concatenate([
          #     base_lin_vel, base_ang_vel, gravity_proj,
          #     joint_pos, joint_vel, joint_torque,
          # ], axis=-1)

          st = self.get_state(base_env)
          ct = self.get_contact(base_env)

          self.env.step(actions)

          if self._step_count == 0:
            mjm = self.env.unwrapped.sim.mj_model

          self.incremental_step += 1
          # print(self.incremental_step)
          self.extract_observation_vectors(base_env, st=st, policy_at=actions, ct=ct)
          if self.incremental_step % self.transitions_per_trajectory == 0 and self.incremental_step != 0:
            self.trajectory_ctr += 1
            self.save_trajectory()
            print(
              f"\n\n\nTrajectory {self.trajectory_ctr} saved to {self.db_path!r}\n"
            )
            self.local_buffer.clear()
          
          if self.trajectory_ctr >= self.buffer_size_d:
            raise ValueError(f'Saved {self.buffer_size_d} trajectories of each trajectory size {self.transitions_per_trajectory}')

        self._step_count += 1
      self._accumulated_sim_time += self._sim_timer.measured_time

  def reset_environment(self) -> None:
    self.env.reset()
    self._step_count = 0
    self._timer.tick()
    # Replay: restart from beginning of trajectory and re-apply first-frame state
    self.idx = 0
    self._replay_initial_state_set = False
    if isinstance(self.trajectory, list) and len(self.trajectory) > 0:
      self._set_replay_initial_state()
      self._replay_initial_state_set = True

  def pause(self) -> None:
    self._is_paused = True
    self._fps_last_frame_time = None
    self.log("[INFO] Simulation paused", VerbosityLevel.INFO)

  def resume(self) -> None:
    self._is_paused = False
    self._timer.tick()
    self._fps_last_frame_time = time.time()
    self.log("[INFO] Simulation resumed", VerbosityLevel.INFO)

  def toggle_pause(self) -> None:
    if self._is_paused:
      self.resume()
    else:
      self.pause()

  def _process_actions(self) -> None:
    """Drain action queue. Runs on the main loop thread."""
    while self._actions:
      action, payload = self._actions.popleft()
      if action == ViewerAction.RESET:
        self.reset_environment()
      elif action == ViewerAction.TOGGLE_PAUSE:
        self.toggle_pause()
      elif action == ViewerAction.SPEED_UP:
        self.increase_speed()
      elif action == ViewerAction.SPEED_DOWN:
        self.decrease_speed()
      else:
        # Hook for subclasses to handle PREV_ENV/NEXT_ENV or CUSTOM actions
        _ = self._handle_custom_action(action, payload)

  def _handle_custom_action(self, action: ViewerAction, payload: Optional[Any]) -> bool:
    del action, payload  # Unused.
    return False

  def tick(self) -> bool:
    self._process_actions()

    elapsed_time = self._timer.tick() * self._time_multiplier
    self._time_until_next_frame -= elapsed_time

    if self._time_until_next_frame > 0:
      return False

    self._time_until_next_frame += self.frame_time
    if self._time_until_next_frame < -self.frame_time:
      self._time_until_next_frame = 0.0

    with self._render_timer.measure_time():
      self.sync_viewer_to_env()
      self.step_simulation()
      self.sync_env_to_viewer()

    self._accumulated_render_time += self._render_timer.measured_time
    self._frame_count += 1
    self._update_fps()

    if self.verbosity >= VerbosityLevel.DEBUG:
      now = time.time()
      if now - self._last_fps_log_time >= 1.0:
        self.log_performance()
        self._last_fps_log_time = now
        self._frame_count = 0
        self._accumulated_sim_time = 0.0
        self._accumulated_render_time = 0.0

    return True

  def run(self, num_steps: Optional[int] = None, db_dir: Optional[str] = None) -> None:
    self._rollout_output_dir = None
    if db_dir is not None:
      n_transitions = self.buffer_size_d * self.transitions_per_trajectory
      folder_name = f"{n_transitions}_transitions"
      self._rollout_output_dir = os.path.join(db_dir, folder_name)
      os.makedirs(self._rollout_output_dir, exist_ok=True)
      print(f"Per-trajectory rollouts directory: {self._rollout_output_dir}")

    self.setup()
    self._last_fps_log_time = time.time()
    self._timer.tick()
    self._fps_last_frame_time = time.time()
    try:
      while self.is_running() and (num_steps is None or self._step_count < num_steps):
        if not self.tick():
          time.sleep(0.001)
    finally:
      self.close()

  def log_performance(self) -> None:
    if self._frame_count > 0:
      avg_sim_ms = self._accumulated_sim_time / self._frame_count * 1000
      avg_render_ms = self._accumulated_render_time / self._frame_count * 1000
      total_ms = avg_sim_ms + avg_render_ms
      status = "PAUSED" if self._is_paused else "RUNNING"
      speed = f"{self._time_multiplier:.1f}x" if self._time_multiplier != 1.0 else "1x"
      print(
        f"[{status}] Step {self._step_count} | FPS: {self._frame_count:.1f} | "
        f"Speed: {speed} | Sim: {avg_sim_ms:.1f}ms | Render: {avg_render_ms:.1f}ms | "
        f"Total: {total_ms:.1f}ms"
      )

  def increase_speed(self) -> None:
    if self._speed_index < len(self.SPEED_MULTIPLIERS) - 1:
      self._speed_index += 1
      self._time_multiplier = self.SPEED_MULTIPLIERS[self._speed_index]

  def decrease_speed(self) -> None:
    if self._speed_index > 0:
      self._speed_index -= 1
      self._time_multiplier = self.SPEED_MULTIPLIERS[self._speed_index]

  def _update_fps(self) -> None:
    if self._is_paused:
      return
    now = time.time()
    if self._fps_last_frame_time is None:
      self._fps_last_frame_time = now
      return
    dt = now - self._fps_last_frame_time
    self._fps_last_frame_time = now
    if dt <= 0:
      return
    self._fps_accum_frames += 1
    self._fps_accum_time += dt
    if self._fps_accum_time >= self._fps_update_interval:
      inst = self._fps_accum_frames / self._fps_accum_time
      if self._smoothed_fps == 0.0:
        self._smoothed_fps = inst
      else:
        self._smoothed_fps = (
          self._fps_alpha * inst + (1.0 - self._fps_alpha) * self._smoothed_fps
        )
      self._fps_accum_frames = 0
      self._fps_accum_time = 0.0
