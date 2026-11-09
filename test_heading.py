"""Verify physical steering, facing direction, and head-relative sensing."""

import math
import unittest

import numpy as np

from obstacle_env import ObstacleAvoidanceEnv


class HeadingTests(unittest.TestCase):
    def make_env(self):
        env = ObstacleAvoidanceEnv(layout="empty", max_steps=7200)
        self.addCleanup(env.close)
        env.reset(seed=42)
        return env

    def command_heading(self, env, heading):
        velocity = 0.2 * np.array([math.cos(heading), math.sin(heading)])
        env.navigator.command = lambda position: velocity.copy()

    def walk(self, env, steps):
        for _ in range(steps):
            _, _, terminated, _, info = env.step(np.zeros(12))
            self.assertFalse(terminated, str(info))
            self.assertLess(abs(info["roll"]), 0.35)
            self.assertLess(abs(info["pitch"]), 0.35)
        return info

    def assert_faces_and_walks(self, env, target):
        self.walk(env, 18 * 60)
        start = env._base_pos()[:2].copy()
        info = self.walk(env, 120)
        delta = env._base_pos()[:2] - start
        direction = math.atan2(delta[1], delta[0])
        self.assertLess(abs(env.wrap_angle(info["yaw"] - target)), math.radians(5))
        self.assertLess(abs(env.wrap_angle(direction - info["yaw"])), math.radians(8))
        self.assertGreater(np.linalg.norm(delta), 0.25)
        self.assertLessEqual(abs(env.gait_yaw_rate), 0.6 + 1e-9)
        return info

    def test_left_turn_faces_and_moves_left(self):
        env = self.make_env()
        self.command_heading(env, math.pi / 4)
        self.assert_faces_and_walks(env, math.pi / 4)

    def test_right_turn_faces_and_moves_right(self):
        env = self.make_env()
        self.command_heading(env, -math.pi / 4)
        self.assert_faces_and_walks(env, -math.pi / 4)

    def test_ninety_degree_direction_change(self):
        env = self.make_env()
        self.command_heading(env, 0.0)
        self.walk(env, 300)
        self.command_heading(env, math.pi / 2)
        self.assert_faces_and_walks(env, math.pi / 2)

    def test_obstacle_is_added_and_sensed_in_front_of_turned_head(self):
        env = self.make_env()
        self.command_heading(env, math.pi / 4)
        info = self.walk(env, 12 * 60)
        before = env._base_pos()[:2].copy()
        env.add_box_ahead()
        delta = np.asarray(env.obstacle_specs[-1]["xy"]) - before
        forward = np.array([math.cos(info["yaw"]), math.sin(info["yaw"])])
        self.assertAlmostEqual(float(np.dot(delta, forward)), 1.8, places=6)
        self.assertAlmostEqual(float(delta[0] * forward[1] - delta[1] * forward[0]), 0.0, places=6)
        env.scan()
        self.assertGreater(len(env.laser_hit_points), 0)
        self.assertLess(env.front_distance, 1.75)
        env.reset(seed=42)
        self.assertLess(abs(env.desired_heading), 0.01)
        self.assertEqual(env.gait_yaw_rate, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
