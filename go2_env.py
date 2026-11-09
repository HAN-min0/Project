"""Laikago walking simulation with a diagonal gait and optional PPO residuals.

The original project calls this Go2GymEnv, so that import is kept. The bundled
PyBullet robot is Laikago, not a physical Unitree Go2. No hardware is controlled.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from pybullet_utils.bullet_client import BulletClient


class Go2GymEnv(gym.Env):
    """60 Hz control, 240 Hz physics, +X forward on a level plane.

    A zero action follows the walking controller. PPO actions add small joint
    corrections to it instead of having to discover a gait from a standing pose.
    The 37-element observation is versioned and intentionally differs from the
    original 32-element policy. Old model ZIPs must not be used with this class.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    ENV_VERSION = "laikago-diagonal-residual-v1"
    CONTROL_DT = 1.0 / 60.0
    DEFAULT_SPEED = 0.25
    LEG_NAMES = ("FR", "FL", "RR", "RL")
    PHASE_OFFSETS = np.array([0.0, 0.5, 0.5, 0.0])

    def __init__(
        self,
        render: bool = False,
        *,
        render_mode: str | None = None,
        target_speed: float = DEFAULT_SPEED,
        max_steps: int = 3600,
    ):
        super().__init__()
        if render and render_mode is None:
            render_mode = "human"
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'.")
        # These are the two tested walk speeds. Zero selects standing.
        if not any(math.isclose(target_speed, v, abs_tol=1e-9) for v in (0.0, 0.25, 0.30)):
            raise ValueError("target_speed must be 0.0 (stand), 0.25, or 0.30 m/s.")
        if not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer.")

        self.render_mode = render_mode
        self.target_speed = float(target_speed)
        self.max_steps = max_steps
        self.time_step = 1.0 / 240.0
        self.frame_skip = 4
        self.control_dt = self.time_step * self.frame_skip
        self.max_force = 45.0
        self.gait_frequency = 2.0
        self.stance_fraction = 0.60
        self.foot_clearance = 0.04
        self.residual_scale = 0.03
        self.stand_pose = np.tile([0.0, 0.07, -0.59], 4)
        self.urdf_path = "laikago/laikago_toes_zup.urdf"
        self.action_space = spaces.Box(-1.0, 1.0, (12,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (37,), dtype=np.float32)

        self._p = BulletClient(connection_mode=p.GUI if render_mode == "human" else p.DIRECT)
        if self._p._client < 0:
            raise RuntimeError("Could not connect to PyBullet.")
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._closed = False
        self._needs_reset = True
        self.robot_id = None
        self.step_counter = 0
        if render_mode == "human":
            self._p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            self._p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

    @property
    def is_connected(self) -> bool:
        return not self._closed and bool(self._p.isConnected())

    @property
    def elapsed_time(self) -> float:
        return self.step_counter * self.control_dt

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._closed:
            raise RuntimeError("This environment has been closed.")
        self._p.resetSimulation()
        # resetSimulation clears physics settings. Reapply them every episode.
        self._p.setTimeStep(self.time_step)
        self._p.setGravity(0, 0, -9.81)
        self._p.setPhysicsEngineParameter(numSolverIterations=80)
        self.plane_id = self._p.loadURDF("plane.urdf")
        # This Z-up URDF uses identity orientation. Euler roll/pitch now measure
        # physical tilt, rather than the old model's 90-degree axis conversion.
        self.robot_id = self._p.loadURDF(self.urdf_path, [0, 0, 0.48], [0, 0, 0, 1])
        joint_info = [self._p.getJointInfo(self.robot_id, i)
                      for i in range(self._p.getNumJoints(self.robot_id))]
        joint_names = {info[1].decode("utf-8"): info[0] for info in joint_info}
        link_names = {info[12].decode("utf-8"): info[0] for info in joint_info}
        suffixes = ("hip_motor_2_chassis_joint", "upper_leg_2_hip_motor_joint",
                    "lower_leg_2_upper_leg_joint")
        self.motor_ids = [joint_names[f"{leg}_{suffix}"]
                          for leg in self.LEG_NAMES for suffix in suffixes]
        self.foot_ids = [link_names[f"toe{leg}"] for leg in self.LEG_NAMES]
        movable = [info[0] for info in joint_info if info[2] != p.JOINT_FIXED]
        if movable != self.motor_ids or len(self.motor_ids) != 12:
            raise RuntimeError("Unexpected Laikago joint order in the installed URDF.")

        for joint_id, angle in zip(self.motor_ids, self.stand_pose):
            self._p.resetJointState(self.robot_id, joint_id, float(angle))
        for link_id in range(-1, len(joint_info)):
            self._p.changeDynamics(
                self.robot_id, link_id, lateralFriction=0.8,
                linearDamping=0.04, angularDamping=0.04,
            )
        # Let all four toes reach the floor before starting the gait.
        for _ in range(300):
            self._set_motor_targets(self.stand_pose)
            self._p.stepSimulation()

        position, orientation = self._p.getBasePositionAndOrientation(self.robot_id)
        self.start_position = np.asarray(position, dtype=np.float64)
        self.reference_height = float(position[2])
        self.reference_y = float(position[1])
        self.nominal_feet = np.array([
            self._p.getLinkState(self.robot_id, foot, computeForwardKinematics=True)[4]
            for foot in self.foot_ids
        ]) - self.start_position
        self.step_counter = 0
        self._previous_action = np.zeros(12)
        self._filtered_action = np.zeros(12)
        self.last_joint_targets = self.stand_pose.copy()
        self._needs_reset = False
        if self.render_mode == "human":
            self.update_camera()
        return self._get_obs(), self._get_info()

    def _set_motor_targets(self, targets):
        self._p.setJointMotorControlArray(
            self.robot_id, self.motor_ids, p.POSITION_CONTROL,
            targetPositions=np.asarray(targets).tolist(),
            forces=[self.max_force] * 12,
            positionGains=[0.3] * 12, velocityGains=[1.0] * 12,
        )

    def _gait_targets(self):
        position, orientation = self._p.getBasePositionAndOrientation(self.robot_id)
        velocity, _ = self._p.getBaseVelocity(self.robot_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
        # Startup is gradual. FR + RL move together; FL + RR are half a cycle out.
        ramp = min(1.0, self.elapsed_time / 1.0)
        stride = self.target_speed * self.stance_fraction / self.gait_frequency * ramp
        targets = self.nominal_feet.copy()
        for leg, offset in enumerate(self.PHASE_OFFSETS):
            phase = (self.elapsed_time * self.gait_frequency + offset) % 1.0
            if phase < self.stance_fraction:
                # A planted foot travels backward relative to the torso.
                forward = (0.5 - phase / self.stance_fraction) * stride
                lift = 0.0
            else:
                swing = (phase - self.stance_fraction) / (1.0 - self.stance_fraction)
                forward = (-0.5 + swing) * stride
                lift = (self.foot_clearance * math.sin(math.pi * swing) * ramp
                        * min(1.0, self.target_speed / 0.15))
            targets[leg, 0] += forward
            targets[leg, 1] += 0.05 * velocity[1]
            targets[leg, 2] += lift

        # Work in the torso frame, then add restoring foot placements. Solving
        # IK against world-level feet directly would instead shorten the low-side
        # legs and preserve a growing torso lean. This sign is intentional.
        targets[:, 2] += 0.5 * (roll * targets[:, 1] - pitch * targets[:, 0])
        heading_correction = 0.5 * yaw
        c, s = math.cos(heading_correction), math.sin(heading_correction)
        yaw_rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        targets = targets @ yaw_rotation.T
        body_rotation = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        world_targets = targets @ body_rotation.T + np.asarray(position)
        world_targets[:, 1] += 0.1 * position[1]
        return np.asarray(self._p.calculateInverseKinematics2(
            self.robot_id, self.foot_ids, world_targets.tolist(),
            solver=p.IK_DLS, maxNumIterations=30, residualThreshold=1e-5,
        ))

    def _apply_action(self, action):
        self._filtered_action += 0.2 * (action - self._filtered_action)
        targets = self._gait_targets() + self.residual_scale * self._filtered_action
        # Software limits are needed because these URDF joints are continuous.
        targets = np.clip(targets, np.tile([-0.8, -0.8, -2.0], 4),
                          np.tile([0.8, 1.4, 0.3], 4))
        self.last_joint_targets = targets
        self._set_motor_targets(targets)

    def _base_pos(self):
        return np.asarray(self._p.getBasePositionAndOrientation(self.robot_id)[0])

    def _get_obs(self):
        states = self._p.getJointStates(self.robot_id, self.motor_ids)
        position, orientation = self._p.getBasePositionAndOrientation(self.robot_id)
        velocity, angular_velocity = self._p.getBaseVelocity(self.robot_id)
        rotation = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        phase = 2.0 * math.pi * self.gait_frequency * self.elapsed_time
        # 12 joint offsets + 12 joint speeds + 3 body linear speeds + 3 body
        # angular speeds + 3 gravity components + 2 phase + 1 speed + 1 height.
        return np.concatenate([
            np.asarray([state[0] for state in states]) - self.stand_pose,
            np.asarray([state[1] for state in states]) * 0.05,
            rotation.T @ velocity,
            rotation.T @ angular_velocity,
            rotation.T @ np.array([0.0, 0.0, -1.0]),
            [math.sin(phase), math.cos(phase), self.target_speed, position[2]],
        ]).astype(np.float32)

    def _get_info(self):
        position, orientation = self._p.getBasePositionAndOrientation(self.robot_id)
        velocity, _ = self._p.getBaseVelocity(self.robot_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
        contacts = self._p.getContactPoints(bodyA=self.robot_id, bodyB=self.plane_id)
        touching = {contact[3] for contact in contacts if contact[9] > 0.0}
        return {
            "env_version": self.ENV_VERSION,
            "elapsed_time": float(self.elapsed_time),
            "distance_x": float(position[0] - self.start_position[0]),
            "lateral_drift": float(position[1] - self.reference_y),
            "forward_velocity": float(velocity[0]),
            "height": float(position[2]),
            "roll": float(roll), "pitch": float(pitch), "yaw": float(yaw),
            "foot_contacts": [foot in touching for foot in self.foot_ids],
            "foot_order": list(self.LEG_NAMES),
        }

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("Call reset() before stepping or after an episode ends.")
        if not self.is_connected:
            raise RuntimeError("PyBullet is no longer connected.")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("action must contain exactly 12 finite values.")
        action = np.clip(action, -1.0, 1.0)
        self._apply_action(action)
        for _ in range(self.frame_skip):
            self._p.stepSimulation()
        self.step_counter += 1
        obs = self._get_obs()
        info = self._get_info()
        _, angular_velocity = self._p.getBaseVelocity(self.robot_id)
        # Velocity is already in m/s. Four physics steps are one 1/60 s action;
        # the old displacement / (1/240 s) overstated forward speed fourfold.
        tracking = math.exp(-((info["forward_velocity"] - self.target_speed) / 0.20) ** 2)
        upright = math.exp(-5.0 * (info["roll"] ** 2 + info["pitch"] ** 2))
        smoothness = float(np.mean((action - self._previous_action) ** 2))
        reward = (
            1.5 * tracking + 0.5 * upright
            - 0.8 * info["lateral_drift"] ** 2
            - 0.2 * info["yaw"] ** 2
            - 0.05 * float(np.dot(angular_velocity, angular_velocity))
            - 0.03 * float(np.mean(action ** 2)) - 0.05 * smoothness
        )
        terminated = bool(
            abs(info["roll"]) > 0.70 or abs(info["pitch"]) > 0.70
            or info["height"] < 0.25 or info["height"] > 0.75
            or not np.isfinite(obs).all()
        )
        truncated = bool(self.step_counter >= self.max_steps)
        if terminated:
            reward -= 5.0
            info["termination_reason"] = "fall_or_invalid_state"
        elif truncated:
            info["termination_reason"] = "time_limit"
        self._previous_action = action.copy()
        self._needs_reset = terminated or truncated
        return obs, float(reward), terminated, truncated, info

    def keyboard_events(self):
        return self._p.getKeyboardEvents() if self.is_connected else {}

    def update_camera(self):
        if self.is_connected and self.robot_id is not None:
            self._p.resetDebugVisualizerCamera(
                cameraDistance=1.8, cameraYaw=45, cameraPitch=-22,
                cameraTargetPosition=self._base_pos().tolist(),
            )

    def render(self):
        if self.robot_id is None:
            raise RuntimeError("Call reset() before rendering.")
        if self.render_mode == "human":
            self.update_camera()
            return None
        if self.render_mode == "rgb_array":
            view = p.computeViewMatrixFromYawPitchRoll(
                self._base_pos().tolist(), 1.5, 45, -20, 0, 2,
            )
            projection = p.computeProjectionMatrixFOV(60, 960 / 640, 0.05, 50)
            camera = self._p.getCameraImage(
                960, 640, viewMatrix=view, projectionMatrix=projection,
                renderer=p.ER_TINY_RENDERER,
            )
            return np.asarray(camera[2], dtype=np.uint8).reshape(640, 960, 4)[:, :, :3]
        return None

    def close(self):
        if not self._closed:
            if self._p.isConnected():
                self._p.disconnect()
            self._closed = True
            self._needs_reset = True
