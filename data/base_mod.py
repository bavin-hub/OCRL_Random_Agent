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


  def extract_and_print_state(self, obs: torch.Tensor, actions: torch.Tensor) -> None:
    t = self._step_count
    mjd = self.env.unwrapped.sim.mj_data
    mjm = self.env.unwrapped.sim.mj_model

    print(f"\n{'='*60}")
    print(f"STATE AT step = {t}")
    print(f"{'='*60}")

    # --- Observation Space ---
    base_lin_vel    = mjd.qvel[0:3]
    base_ang_vel    = mjd.qvel[3:6]
    
    quat = mjd.qpos[3:7]
    qw, qx, qy, qz = quat
    gravity_proj = np.array([
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz)
    ])

    # gravity_proj    = mjd.qpos[3:7]  # quaternion — project if needed
    joint_pos       = mjd.qpos[7:]
    joint_vel       = mjd.qvel[6:]
    joint_torque    = mjd.qfrc_actuator[6:]

    # # for 23 dof
    # dof_indices = list(range(0, 20)) + list(range(23, 27))
    # joint_pos    = mjd.qpos[7:][dof_indices]
    # joint_vel    = mjd.qvel[6:][dof_indices]
    # joint_torque = mjd.qfrc_actuator[6:][dof_indices]
    


    # print(f"\n[Observation Space]")
    # print(f"  base linear velocity  (0:3)  : {np.round(base_lin_vel, 4)}")
    # print(f"  base angular velocity (3:6)  : {np.round(base_ang_vel, 4)}")
    # print(f"  projected gravity     (6:9)  : {np.round(gravity_proj, 4)}")
    # print(f"  joint positions       (9:38) : {np.round(joint_pos, 4)}")
    # print(f"  joint velocities      (38:67): {np.round(joint_vel, 4)}")
    # print(f"  joint torques         (67:96): {np.round(joint_torque, 4)}")

    # --- Action Space ---
    action_np = actions.cpu().numpy().squeeze()
    # # for 23 dof
    # action_23    = actions.cpu().numpy().squeeze()[dof_indices]
    # print(f"\n[Action Space]")
    # print(f"  joint pos targets q*  (0:29) : {np.round(action_np, 4)}")

    # --- Privileged / Contact Info ---
    # Foot body IDs — adjust if needed after checking mjm body names
    foot_body_ids = [
    mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "robot/left_ankle_roll_link"),
    mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "robot/right_ankle_roll_link"),
    ]
    foot_geom_ids = [i for i in range(mjm.ngeom) if mjm.geom_bodyid[i] in foot_body_ids]
    
    left_geoms  = [i for i in range(mjm.ngeom) if mjm.geom_bodyid[i] == foot_body_ids[0]]
    right_geoms = [i for i in range(mjm.ngeom) if mjm.geom_bodyid[i] == foot_body_ids[1]]
    foot_heights = np.array([
        np.min([mjd.geom_xpos[g, 2] for g in left_geoms]),
        np.min([mjd.geom_xpos[g, 2] for g in right_geoms]),
    ])
    # foot_heights    = np.array([mjd.geom_xpos[gid, 2] for gid in foot_geom_ids])
    
    foot_velocities = np.array([np.linalg.norm(mjd.cvel[bid, 0:3]) for bid in foot_body_ids])

    contact_flags = np.zeros(mjm.nbody, dtype=int)
    for i in range(mjd.ncon):
        con = mjd.contact[i]
        contact_flags[mjm.geom_bodyid[con.geom1]] = 1
        contact_flags[mjm.geom_bodyid[con.geom2]] = 1

    body_contact = contact_flags[:26]
    # for 23 dof
    body_indices = list(range(0, 22)) + list(range(25, 29))  # 26 bodies, skipping wrists
    body_contact = contact_flags[body_indices]
    

    # print(f"\n[Privileged / Critic Info]")
    # print(f"  body contact          (0:26) : {body_contact}")
    # print(f"  foot height           (26:28): {np.round(foot_heights, 4)}")
    # print(f"  foot velocity         (28:30): {np.round(foot_velocities, 4)}")
    print(base_lin_vel.shape)
    print(base_ang_vel.shape)
    print(gravity_proj.shape)
    print(joint_pos.shape)
    print(joint_vel.shape)
    print(joint_torque.shape)
    print(body_contact.shape)
    print(foot_heights.shape)
    print(foot_velocities.shape)
    print(action_np.shape)
    print(f"{'='*60}\n\n\n")

  

  # Core loop.

  def step_simulation(self) -> None:
    if self._is_paused:
      return
    # Wrap in no_grad mode to prevent gradient accumulation and memory leaks.
    # NOTE: Using torch.inference_mode() causes a "RuntimeError: Inplace update to
    # inference tensor outside InferenceMode is not allowed" inside the command
    # manager when resetting the env with a key callback.
    with torch.no_grad():
      with self._sim_timer.measure_time():
        obs = self.env.get_observations()
        actions = self.policy(obs)
        self.env.step(actions)
        if self._step_count == 0:
          mjm = self.env.unwrapped.sim.mj_model
          # for i in range(mjm.nbody):
          #     print(i, mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, i))
          # for i in range(mjm.njnt):
          #   print(i, mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, i))
        self.extract_and_print_state(obs, actions)
        self._step_count += 1
      self._accumulated_sim_time += self._sim_timer.measured_time

  def reset_environment(self) -> None:
    self.env.reset()
    self._step_count = 0
    self._timer.tick()

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

  def run(self, num_steps: Optional[int] = None, db_path='') -> None:
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
