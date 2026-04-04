"""Base class for environment viewers.

The viewer runs a policy against an RL environment and displays the
result in real time.

The Problem
===========

Each call to ``env.step()`` advances the simulation by a fixed amount
of sim time (``step_dt``, set by the env config). The simplest viewer
would call ``env.step()`` once per loop iteration, but iterations take
however long the hardware needs. A fast machine loops 67 times per
second and the simulation runs too fast; a slow machine loops 30 times
and it runs too slow. Playback speed would depend on hardware.

The viewer needs to answer a different question: given how much real
time just elapsed, how many calls to ``env.step()`` will keep the
simulation advancing at the right pace?

Budget Accumulator
==================

A single variable, ``_sim_budget``, tracks how much sim time has
accumulated but not yet been simulated. Each tick of the main loop:

  1. Measure real time elapsed since the last tick.
  2. Multiply by the speed setting and add to the budget.
  3. Call ``env.step()`` in a loop, subtracting ``step_dt`` from the
     budget each time, until the budget is less than one step.
  4. Carry the leftover to the next tick.

Example at 1x speed, step_dt = 0.02s (50 Hz control), 60 fps::

  tick 1:  +0.0167s  ->  budget = 0.0167  ->  no step (< 0.02)
  tick 2:  +0.0167s  ->  budget = 0.0334  ->  1 step, 0.0134 left
  tick 3:  +0.0167s  ->  budget = 0.0301  ->  1 step, 0.0101 left
  ...

This averages to 50 steps per second on any hardware. At 2x speed the
elapsed time is doubled before adding to the budget, so steps happen
twice as often. At 0.5x, half as often.

Rendering is independent: the display refreshes at ``frame_rate``
(e.g. 60 Hz) whether or not a new step happened. Some frames will
re-display the same state.

If physics is too slow to keep up, the budget grows without bound. A
real time deadline (one frame period) caps each burst so the renderer
always gets a turn. Leftover budget is dropped and ``_was_capped`` is
set.

Main Loop
=========

::

  run()
    setup()
    while running:
      tick()                    -> True if a frame was produced
        _process_actions()      drain UI action queue
        _step_physics(dt)       accumulate budget, step until spent
        sync_env_to_viewer()    push state to display (at frame_rate)
      sleep(1ms)                yield CPU when no frame is due

Subclasses implement setup(), sync_env_to_viewer(), sync_viewer_to_env(),
close(), and is_running().
"""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Any, Optional, Protocol
import os, json, sqlite3
import numpy as np
import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg


class VerbosityLevel(IntEnum):
  SILENT = 0
  INFO = 1
  DEBUG = 2


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


@dataclass(frozen=True)
class ViewerStatus:
  paused: bool
  step_count: int
  speed_multiplier: float
  speed_label: str
  target_realtime: float
  actual_realtime: float
  smoothed_fps: float
  capped: bool
  last_error: str | None


class ViewerAction(Enum):
  RESET = "reset"
  TOGGLE_PAUSE = "toggle_pause"
  SINGLE_STEP = "single_step"
  RESET_SPEED = "reset_speed"
  SPEED_UP = "speed_up"
  SPEED_DOWN = "speed_down"
  PREV_ENV = "prev_env"
  NEXT_ENV = "next_env"
  TOGGLE_PLOTS = "toggle_plots"
  TOGGLE_DEBUG_VIS = "toggle_debug_vis"
  TOGGLE_SHOW_ALL_ENVS = "toggle_show_all_envs"
  CUSTOM = "custom"


class BaseViewer(ABC):
  """Abstract base class for environment viewers."""

  SPEED_MULTIPLIERS = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 8.0]

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

    # State.
    self._is_paused = False
    self._step_count = 0
    self._last_error: str | None = None

    # Speed.
    self._speed_index = self.SPEED_MULTIPLIERS.index(8.0)
    self._time_multiplier = self.SPEED_MULTIPLIERS[self._speed_index]

    # Physics accumulator and render timer.
    self._sim_budget = 0.0
    self._time_until_next_render = 0.0
    self._last_tick_time = 0.0
    self._was_capped = False

    # Windowed stats, updated every 0.5s.
    self._stats_frames = 0
    self._stats_steps = 0
    self._stats_last_time = 0.0
    self._fps = 0.0
    self._sps = 0.0

    # changes starts #
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
    # change ends #

    # Action queue, drained on main thread each tick.
    self._actions: deque[tuple[ViewerAction, Optional[Any]]] = deque()

  # Abstract hooks.

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

  def _forward_paused(self) -> None:  # noqa: B027
    """Hook for subclasses to run forward kinematics while paused."""

  def _handle_custom_action(self, action: ViewerAction, payload: Optional[Any]) -> bool:
    del action, payload
    return False

  # Logging.

  def log(self, message: str, level: VerbosityLevel = VerbosityLevel.INFO) -> None:
    if self.verbosity >= level:
      print(message)

  # Thread-safe action requests.

  def request_reset(self) -> None:
    self._actions.append((ViewerAction.RESET, None))

  def request_toggle_pause(self) -> None:
    self._actions.append((ViewerAction.TOGGLE_PAUSE, None))

  def request_single_step(self) -> None:
    self._actions.append((ViewerAction.SINGLE_STEP, None))

  def request_speed_up(self) -> None:
    self._actions.append((ViewerAction.SPEED_UP, None))

  def request_speed_down(self) -> None:
    self._actions.append((ViewerAction.SPEED_DOWN, None))

  def request_reset_speed(self) -> None:
    self._actions.append((ViewerAction.RESET_SPEED, None))

  def request_action(self, name: str, payload: Optional[Any] = None) -> None:
    try:
      action = ViewerAction[name]
    except KeyError:
      action = ViewerAction.CUSTOM
    self._actions.append((action, payload))

  # Speed controls.

  def increase_speed(self) -> None:
    if self._speed_index < len(self.SPEED_MULTIPLIERS) - 1:
      self._speed_index += 1
      self._time_multiplier = self.SPEED_MULTIPLIERS[self._speed_index]

  def decrease_speed(self) -> None:
    if self._speed_index > 0:
      self._speed_index -= 1
      self._time_multiplier = self.SPEED_MULTIPLIERS[self._speed_index]

  def reset_speed(self) -> None:
    self._speed_index = self.SPEED_MULTIPLIERS.index(1.0)
    self._time_multiplier = 1.0

  # Pause and resume.

  def pause(self) -> None:
    self._is_paused = True
    self.log("[INFO] Simulation paused", VerbosityLevel.INFO)

  def resume(self) -> None:
    self._is_paused = False
    self._last_error = None
    self._sim_budget = 0.0
    self._last_tick_time = time.perf_counter()
    self.log("[INFO] Simulation resumed", VerbosityLevel.INFO)

  def toggle_pause(self) -> None:
    if self._is_paused:
      self.resume()
    else:
      self.pause()

  # change starts #

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


  def normalize_policy_vec(self, command: np.ndarray, base_lin_vel_limit: float) -> np.ndarray:
    """
    Normalize the velocity command vector.

    Args:
        command (np.ndarray): shape (3,) velocity command
        base_lin_vel_limit (float): max absolute linear velocity

    Returns:
        np.ndarray: normalized velocity command
    """
    # Normalize each component (symmetric around 0)
    command_n = self._normalize_symmetric(command, base_lin_vel_limit)

    return command_n  # shape (3,)
  
  def extract_observation_vectors(self, env, st, policy_at, ct, command, env_id=0):
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
    
    policy_vec = command
    
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

    policy_vec_n = self.normalize_policy_vec(
        policy_vec,
        base_lin_vel_limit=6.0,
    )

    # Append to buffer
    self.local_buffer.append((
        st_n.tolist(),
        contact_vec.tolist(),  # use the newly computed contact vector
        target_n.tolist(),
        policy_at.cpu().numpy().squeeze().tolist(),
        policy_vec_n.tolist()
    ))

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

  # change ends #

  # Core loop.

  def _execute_step(self) -> bool:
    """Run one obs/policy/step cycle.

    Returns True on success, False if step failed.
    """
    try:
      with torch.no_grad():

        # change starts #
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
        # change ends #

        obs = self.env.get_observations()
        actions = self.policy(obs)

        # change starts #
        st = self.get_state(base_env)
        ct = self.get_contact(base_env)
        critic_obs = obs["critic"]  # shape: [1, 113]

        # Extract command (3D vector)
        command = critic_obs[:, 6:9]  # shape: [1, 3]

        # If you want as numpy array
        command_np = command.cpu().numpy().squeeze(0)  # shape: (3,)
                # change ends

        self.env.step(actions)

        # change starts #
        self.incremental_step += 1
        self.extract_observation_vectors(base_env, st=st, policy_at=actions, ct=ct, command=command_np)
        if self.incremental_step % self.transitions_per_trajectory == 0 and self.incremental_step != 0:
            self.trajectory_ctr += 1
            self.save_trajectory()
            print(
              f"\n\nTrajectory {self.trajectory_ctr} saved to {self.db_path!r}\n"
            )
            self.local_buffer.clear()
        if self.trajectory_ctr >= self.buffer_size_d:
            raise ValueError(f'Saved {self.buffer_size_d} trajectories of each trajectory size {self.transitions_per_trajectory}')
        # change ends #

        self._step_count += 1
        self._stats_steps += 1
        return True
    except Exception:
      self._last_error = traceback.format_exc()
      self.log(
        f"[ERROR] Exception during step:\n{self._last_error}",
        VerbosityLevel.SILENT,
      )
      self.pause()
      return False

  def _step_physics(self, dt: float) -> None:
    """Run physics steps for this frame's sim-time budget."""
    step_dt = self.env.unwrapped.step_dt
    self._sim_budget += dt * self._time_multiplier
    self._was_capped = False

    if self._sim_budget < step_dt:
      return

    self.sync_viewer_to_env()
    hit_deadline = False
    deadline = time.perf_counter() + self.frame_time
    while self._sim_budget >= step_dt:
      if not self._execute_step():
        self._sim_budget = 0.0
        return
      self._sim_budget -= step_dt
      if time.perf_counter() > deadline:
        hit_deadline = True
        break

    if hit_deadline:
      # Only report capped if we actually had to drop remaining work. A transient stall
      # (GC pause) during a single step triggers the deadline but leaves no remaining
      # budget, so it's not a real cap.
      self._was_capped = self._sim_budget >= step_dt
      self._sim_budget = min(self._sim_budget, step_dt)

  def _single_step(self) -> None:
    """Advance exactly one step while paused."""
    if not self._is_paused:
      return
    self.sync_viewer_to_env()
    self._execute_step()

  def reset_environment(self) -> None:
    self.env.reset()
    reset_fn = getattr(self.policy, "reset", None)
    if reset_fn is not None:
      reset_fn()
    self._step_count = 0
    self._sim_budget = 0.0
    self._last_error = None
    self._last_tick_time = time.perf_counter()

  def _process_actions(self) -> None:
    """Drain action queue. Runs on the main loop thread."""
    while self._actions:
      action, payload = self._actions.popleft()
      if action == ViewerAction.RESET:
        self.reset_environment()
      elif action == ViewerAction.TOGGLE_PAUSE:
        self.toggle_pause()
      elif action == ViewerAction.SINGLE_STEP:
        self._single_step()
      elif action == ViewerAction.RESET_SPEED:
        self.reset_speed()
      elif action == ViewerAction.SPEED_UP:
        self.increase_speed()
      elif action == ViewerAction.SPEED_DOWN:
        self.decrease_speed()
      else:
        _ = self._handle_custom_action(action, payload)

  def tick(self) -> bool:
    """Advance one tick: drain actions, step physics, maybe render.

    Returns True when a render frame was produced, False otherwise.
    """
    now = time.perf_counter()
    dt = now - self._last_tick_time
    self._last_tick_time = now

    self._process_actions()

    if self._is_paused:
      self._forward_paused()
    else:
      self._step_physics(dt)

    # Render at fixed frame rate.
    self._time_until_next_render -= dt
    if self._time_until_next_render > 0:
      return False

    self._time_until_next_render += self.frame_time
    if self._time_until_next_render < -self.frame_time:
      self._time_until_next_render = 0.0

    self.sync_env_to_viewer()
    self._stats_frames += 1
    return True

  def run(self, num_steps: Optional[int] = None, db_dir: Optional[str] = None) -> None:
    # change starts #
    self._rollout_output_dir = None
    if db_dir is not None:
      n_transitions = self.buffer_size_d * self.transitions_per_trajectory
      folder_name = f"{n_transitions}_transitions"
      self._rollout_output_dir = os.path.join(db_dir, folder_name)
      os.makedirs(self._rollout_output_dir, exist_ok=True)
      print(f"Per-trajectory rollouts directory: {self._rollout_output_dir}")
    # change ends #

    self.setup()
    now = time.perf_counter()
    self._stats_last_time = now
    self._last_tick_time = now
    try:
      while self.is_running() and (num_steps is None or self._step_count < num_steps):
        if not self.tick():
          time.sleep(0.001)
        self._update_stats()
    finally:
      self.close()

  # Stats.

  def _update_stats(self) -> None:
    if self._is_paused:
      return
    now = time.perf_counter()
    dt = now - self._stats_last_time
    if dt >= 0.5:
      self._fps = self._stats_frames / dt
      self._sps = self._stats_steps / dt
      self._stats_frames = 0
      self._stats_steps = 0
      self._stats_last_time = now

      if self.verbosity >= VerbosityLevel.DEBUG:
        status = self.get_status()
        print(
          f"[{'PAUSED' if status.paused else 'RUNNING'}] "
          f"Step {status.step_count} | "
          f"FPS: {status.smoothed_fps:.0f} | "
          f"Speed: {status.speed_label} | "
          f"RTF: {status.actual_realtime:.2f}x / "
          f"{status.target_realtime:.2f}x"
        )

  @property
  def target_realtime(self) -> float:
    return self._time_multiplier

  @property
  def actual_realtime(self) -> float:
    return self._sps * self.env.unwrapped.step_dt

  @staticmethod
  def _format_speed(multiplier: float) -> str:
    if multiplier == 1.0:
      return "1x"
    inv = 1.0 / multiplier
    inv_rounded = round(inv)
    if abs(inv - inv_rounded) < 1e-9 and inv_rounded > 0:
      return f"1/{inv_rounded}x"
    return f"{multiplier:.3g}x"

  def get_status(self) -> ViewerStatus:
    return ViewerStatus(
      paused=self._is_paused,
      step_count=self._step_count,
      speed_multiplier=self._time_multiplier,
      speed_label=self._format_speed(self._time_multiplier),
      target_realtime=self.target_realtime,
      actual_realtime=self.actual_realtime,
      smoothed_fps=self._fps,
      capped=self._was_capped,
      last_error=self._last_error,
    )

