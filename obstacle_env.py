"""Physical obstacle avoidance with forward walking and whole-body steering."""

from __future__ import annotations

import math

import numpy as np
import pybullet as p

from go2_env import Go2GymEnv
from navigation import RangeNavigator


class ObstacleAvoidanceEnv(Go2GymEnv):
    NAV_VERSION = "range-astar-heading-v2"
    LASER_RANGE = 2.8
    LASER_RAYS = 180
    LASER_HEIGHTS = (0.08, 0.28, 0.50)

    def __init__(self, *args, layout="course", avoidance=True, **kwargs):
        if layout not in ("course", "single", "blocked", "empty"):
            raise ValueError("layout must be course, single, blocked, or empty.")
        self.layout = layout
        self.avoidance = bool(avoidance)
        self.obstacle_ids = []
        self.obstacle_specs = []
        self.navigator = None
        self.requested_velocity = np.zeros(2)
        self.gait_velocity = np.zeros(2)
        self._velocity_bias = np.zeros(2)
        self.desired_heading = 0.0
        self.path_heading = 0.0
        self.gait_yaw_rate = 0.0
        self._was_walking = False
        self.laser_hit_points = np.empty((0, 3))
        self.laser_results = []
        self.laser_starts = []
        self.laser_ends = []
        self.front_distance = self.LASER_RANGE
        self.collision_steps = 0
        self.min_clearance = math.inf
        self.goal = np.array([9.5, 0.0])
        super().__init__(*args, **kwargs)

    def reset(self, *, seed=None, options=None):
        self.obstacle_ids = []
        self.obstacle_specs = []
        self.requested_velocity = np.zeros(2)
        self.gait_velocity = np.zeros(2)
        self._velocity_bias = np.zeros(2)
        self.gait_yaw_rate = 0.0
        self._was_walking = False
        self.collision_steps = 0
        self.min_clearance = math.inf
        observation, _ = super().reset(seed=seed, options=options)
        _, orientation = self._p.getBasePositionAndOrientation(self.robot_id)
        self.desired_heading = self.path_heading = p.getEulerFromQuaternion(orientation)[2]
        # Physical collisions remain enabled. Bit 2 merely lets lidar exclude
        # the robot and the floor, instead of measuring the robot's own legs.
        for link in range(-1, self._p.getNumJoints(self.robot_id)):
            self._p.setCollisionFilterGroupMask(self.robot_id, link, 1, 3)
        self._p.setCollisionFilterGroupMask(self.plane_id, -1, 1, 3)
        self.navigator = RangeNavigator(goal=self.goal)
        self._build_course()
        self.scan()
        self.navigator.observe(self.laser_hit_points)
        self.navigator.plan(self._base_pos()[:2])
        self._debug_ids = []
        return observation, self.navigation_info(super()._get_info())

    def _build_course(self):
        if self.layout == "single":
            self.add_box((2.4, 0.0), (0.60, 0.85, 0.65))
        elif self.layout == "blocked":
            self.add_box((2.4, 0.0), (0.5, 7.0, 0.75))
        elif self.layout == "course":
            # Scene generation uses a seed; the navigator receives only scans.
            for i, x in enumerate((2.4, 5.0, 7.6)):
                y = (0.0, 0.55, -0.55)[i] + self.np_random.uniform(-0.12, 0.12)
                width = self.np_random.uniform(0.65, 0.95)
                depth = self.np_random.uniform(0.45, 0.65)
                self.add_box((x, y), (depth, width, 0.65))
        self._draw_goal()

    def add_box(self, xy, size=(0.55, 0.75, 0.65)):
        xy = np.asarray(xy, dtype=float)
        size = np.asarray(size, dtype=float)
        if xy.shape != (2,) or size.shape != (3,) or not np.isfinite(xy).all() or not np.isfinite(size).all() or (size <= 0).any():
            raise ValueError("Box needs finite XY and three positive dimensions.")
        collision = self._p.createCollisionShape(p.GEOM_BOX, halfExtents=(size / 2).tolist())
        colors = ((0.9, 0.25, 0.10, 1), (0.95, 0.62, 0.08, 1), (0.50, 0.28, 0.78, 1))
        visual = self._p.createVisualShape(p.GEOM_BOX, halfExtents=(size / 2).tolist(),
                                          rgbaColor=colors[len(self.obstacle_ids) % len(colors)])
        obstacle = self._p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual,
            basePosition=[float(xy[0]), float(xy[1]), float(size[2] / 2)],
        )
        self._p.setCollisionFilterGroupMask(obstacle, -1, 2, 3)
        self.obstacle_ids.append(obstacle)
        self.obstacle_specs.append({"xy": xy.tolist(), "size": size.tolist()})
        return obstacle

    def add_box_ahead(self):
        position = self._base_pos()[:2]
        _, orientation = self._p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(orientation)[2]
        direction = np.array([math.cos(yaw), math.sin(yaw)])
        return self.add_box(position + 1.8 * direction)

    def scan(self):
        position, orientation = self._p.getBasePositionAndOrientation(self.robot_id)
        xy = np.asarray(position[:2])
        yaw = p.getEulerFromQuaternion(orientation)[2]
        c, s = math.cos(yaw), math.sin(yaw)
        world_to_body = np.array([[c, s], [-s, c]])
        angles = np.arange(self.LASER_RAYS) * (2 * math.pi / self.LASER_RAYS)
        unit = np.column_stack((np.cos(angles), np.sin(angles)))
        starts, ends = [], []
        for height in self.LASER_HEIGHTS:
            for direction in unit:
                starts.append([float(xy[0]), float(xy[1]), height])
                end = xy + self.LASER_RANGE * direction
                ends.append([float(end[0]), float(end[1]), height])
        results = self._p.rayTestBatch(starts, ends, collisionFilterMask=2)
        self.laser_starts, self.laser_ends = starts, ends
        self.laser_results = results
        self.laser_hit_points = np.array([hit[3] for hit in results if hit[0] >= 0], dtype=float).reshape(-1, 3)
        front = []
        for hit in results:
            if hit[0] >= 0:
                delta = world_to_body @ (np.asarray(hit[3][:2]) - xy)
                if delta[0] > 0 and abs(delta[1]) < 0.40:
                    front.append(float(delta[0]))
        self.front_distance = min(front, default=self.LASER_RANGE)

    @staticmethod
    def wrap_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _gait_targets(self):
        position, orientation = self._p.getBasePositionAndOrientation(self.robot_id)
        velocity, angular_velocity = self._p.getBaseVelocity(self.robot_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
        c, s = math.cos(yaw), math.sin(yaw)
        body_velocity = np.array([[c, s], [-s, c]]) @ np.asarray(velocity[:2])
        speed = float(np.linalg.norm(self.requested_velocity))
        walking = speed > 0.02
        if walking:
            self.path_heading = math.atan2(self.requested_velocity[1], self.requested_velocity[0])
            # Filter grid-path changes, then limit the rotation of the reference.
            change = (1.0 - math.exp(-self.control_dt / 0.25)) * self.wrap_angle(self.path_heading - self.desired_heading)
            self.desired_heading = self.wrap_angle(self.desired_heading + float(np.clip(
                change, -0.8 * self.control_dt, 0.8 * self.control_dt,
            )))
        elif self._was_walking:
            # Hold the arrival orientation rather than turning back to world +X.
            self.desired_heading = self.path_heading = yaw
        self._was_walking = walking
        heading_error = self.wrap_angle(self.desired_heading - yaw)
        path_error = self.wrap_angle(self.path_heading - yaw)
        # An abrupt direction change first slows forward motion. The requested
        # body-sideways velocity is zero: the torso and fixed head face the walk.
        forward_speed = speed * max(0.0, math.cos(path_error)) ** 2
        body_request = np.array([forward_speed, 0.0])
        error = body_request - body_velocity
        if walking:
            self._velocity_bias = np.clip(self._velocity_bias + 0.3 * error * self.control_dt,
                                          [-0.12, -0.08], [0.12, 0.08])
        else:
            self._velocity_bias *= 0.96
        desired = body_request + 0.5 * error + self._velocity_bias
        length = np.linalg.norm(desired)
        desired *= min(1.0, 0.30 / max(length, 1e-9))
        change = desired - self.gait_velocity
        self.gait_velocity += change * min(1.0, 0.15 * self.control_dt / max(np.linalg.norm(change), 1e-9))
        yaw_request = float(np.clip(1.8 * heading_error - 0.25 * angular_velocity[2], -0.6, 0.6)) if walking else 0.0
        self.gait_yaw_rate += float(np.clip(yaw_request - self.gait_yaw_rate,
                                            -self.control_dt, self.control_dt))
        ramp = min(1.0, self.elapsed_time)
        activity = float(np.linalg.norm(self.gait_velocity)) + 0.35 * abs(self.gait_yaw_rate)
        targets = self.nominal_feet.copy()
        for leg, offset in enumerate(self.PHASE_OFFSETS):
            x, y = self.nominal_feet[leg, :2]
            # Differential foot paths create physical yaw torque through ground
            # contacts. No base-pose reset or external body torque is used.
            foot_velocity = self.gait_velocity + self.gait_yaw_rate * np.array([-y, x])
            stride = foot_velocity * self.stance_fraction / self.gait_frequency * ramp
            phase = (self.elapsed_time * self.gait_frequency + offset) % 1.0
            if phase < self.stance_fraction:
                delta = (0.5 - phase / self.stance_fraction) * stride
                lift = 0.0
            else:
                swing = (phase - self.stance_fraction) / (1.0 - self.stance_fraction)
                delta = (-0.5 + swing) * stride
                lift = self.foot_clearance * math.sin(math.pi * swing) * ramp * min(1.0, activity / 0.15)
            targets[leg, :2] += delta
            targets[leg, 1] += 0.05 * body_velocity[1]
            targets[leg, 2] += lift
        targets[:, 2] += 0.5 * (roll * targets[:, 1] - pitch * targets[:, 0])
        correction = 0.3 * self.wrap_angle(yaw - self.desired_heading)
        c, s = math.cos(correction), math.sin(correction)
        targets = targets @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T
        rotation = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        world = targets @ rotation.T + position
        return np.asarray(self._p.calculateInverseKinematics2(
            self.robot_id, self.foot_ids, world.tolist(), maxNumIterations=30,
            residualThreshold=1e-5,
        ))

    def navigation_info(self, info):
        info.update(
            nav_version=self.NAV_VERSION, layout=self.layout,
            nav_mode=self.navigator.mode if self.navigator else "SCANNING",
            goal=self.goal.tolist(),
            goal_distance=float(np.linalg.norm(self.goal - self._base_pos()[:2])),
            goal_reached=bool(self.navigator and self.navigator.goal_reached),
            front_distance=float(self.front_distance),
            command_velocity=self.requested_velocity.tolist(),
            desired_heading=float(self.desired_heading), path_heading=float(self.path_heading),
            commanded_yaw_rate=float(self.gait_yaw_rate),
            replans=self.navigator.replans if self.navigator else 0,
            observed_cells=self.navigator.seen_hit_count if self.navigator else 0,
            collision_steps=self.collision_steps,
            min_obstacle_clearance=None if not math.isfinite(self.min_clearance) else self.min_clearance,
        )
        return info

    def step(self, action):
        if not self._needs_reset:
            if self.step_counter % 6 == 0:
                self.scan()
            if self.step_counter % 12 == 0:
                self.navigator.observe(self.laser_hit_points)
                self.navigator.plan(self._base_pos()[:2])
            self.requested_velocity = (self.navigator.command(self._base_pos()[:2]) if self.avoidance
                                       else np.array([0.20, 0.0]))
        observation, reward, terminated, truncated, info = super().step(action)
        collided = False
        for obstacle in self.obstacle_ids:
            if self._p.getContactPoints(bodyA=self.robot_id, bodyB=obstacle):
                collided = True
            # This is reporting only. The planner never reads exact geometry.
            closest = self._p.getClosestPoints(self.robot_id, obstacle, distance=0.8)
            if closest:
                self.min_clearance = min(self.min_clearance, min(point[8] for point in closest))
        if collided:
            self.collision_steps += 1
            terminated = True
            self._needs_reset = True
            info["termination_reason"] = "obstacle_collision"
        return observation, reward, terminated, truncated, self.navigation_info(info)

    def _draw_goal(self):
        # Visual markers do not have collision shapes.
        visual = self._p.createVisualShape(p.GEOM_CYLINDER, radius=0.24, length=0.006,
                                          rgbaColor=[0.08, 0.80, 0.25, 1])
        self._p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual,
                               basePosition=[float(self.goal[0]), float(self.goal[1]), 0.004])

    def update_navigation_display(self, show_rays=False):
        if self.render_mode != "human" or not self.is_connected:
            return
        self._p.removeAllUserDebugItems()
        pos = self._base_pos()
        _, orientation = self._p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(orientation)[2]
        text = f"AUTO: {self.navigator.mode} | heading {math.degrees(yaw):+.0f}deg | front {self.front_distance:.2f}m"
        self._p.addUserDebugText(text, [pos[0] - 0.3, pos[1], 1.0], [0.05, 0.12, 0.2], 1.2)
        path = self.navigator.path
        for first, second in zip(path, path[1:]):
            self._p.addUserDebugLine([*first, 0.025], [*second, 0.025], [0, 0.65, 0.8], 2)
        if show_rays:
            offset = self.LASER_RAYS
            for index in range(offset, offset + self.LASER_RAYS, 8):
                result = self.laser_results[index]
                end = result[3] if result[0] >= 0 else self.laser_ends[index]
                color = [1, 0.2, 0.1] if result[0] >= 0 else [0.2, 0.8, 0.3]
                self._p.addUserDebugLine(self.laser_starts[index], end, color, 1)

    def update_camera(self):
        if self.is_connected and self.robot_id is not None:
            pos = self._base_pos()
            self._p.resetDebugVisualizerCamera(3.0, 45, -28, [pos[0] + 0.45, pos[1], 0.25])

    def render(self):
        if self.render_mode != "rgb_array":
            return super().render()
        if self.robot_id is None:
            raise RuntimeError("Call reset() before rendering.")
        pos = self._base_pos()
        view = p.computeViewMatrixFromYawPitchRoll([pos[0] + 0.5, pos[1], 0.25],
                                                  2.7, 45, -30, 0, 2)
        projection = p.computeProjectionMatrixFOV(60, 960 / 640, 0.05, 50)
        camera = self._p.getCameraImage(960, 640, viewMatrix=view, projectionMatrix=projection,
                                       renderer=p.ER_TINY_RENDERER)
        return np.asarray(camera[2], dtype=np.uint8).reshape(640, 960, 4)[:, :, :3]
