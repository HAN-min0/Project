"""End-to-end obstacle avoidance tests using real PyBullet contacts and motion."""

import unittest

import numpy as np

from obstacle_env import ObstacleAvoidanceEnv


class ObstacleTests(unittest.TestCase):
    def make_env(self, **kwargs):
        env = ObstacleAvoidanceEnv(max_steps=7200, **kwargs)
        self.addCleanup(env.close)
        env.reset(seed=42)
        return env

    def walk_to_goal(self, env, seconds=110):
        info = None
        for _ in range(seconds * 60):
            _, _, terminated, _, info = env.step(np.zeros(12))
            self.assertFalse(terminated, f"Failed at {env.elapsed_time:.2f}s: {info}")
            if info["goal_reached"]:
                break
        self.assertTrue(info["goal_reached"], f"Goal was not reached: {info}")
        self.assertEqual(info["collision_steps"], 0)
        self.assertLess(info["goal_distance"], 0.20)
        return info

    def test_lidar_detects_new_low_box_and_excludes_robot_and_floor(self):
        env = self.make_env(layout="empty")
        self.assertEqual(len(env.laser_hit_points), 0)
        env.add_box((1.6, 0), (0.5, 0.8, 0.12))
        env.scan()
        self.assertGreater(len(env.laser_hit_points), 0)
        self.assertAlmostEqual(env.front_distance, 1.35 - env.start_position[0], delta=0.03)

    def test_three_obstacles_reached_and_goal_stops(self):
        env = self.make_env(layout="course")
        info = self.walk_to_goal(env)
        self.assertGreater(info["observed_cells"], 20)
        self.assertGreater(info["min_obstacle_clearance"], 0.10)
        before = env._base_pos()[:2].copy()
        for _ in range(180):
            _, _, terminated, _, info = env.step(np.zeros(12))
            self.assertFalse(terminated)
        self.assertEqual(info["nav_mode"], "GOAL_REACHED")
        self.assertLess(np.linalg.norm(env._base_pos()[:2] - before), 0.20)
        self.assertLess(np.linalg.norm(env._p.getBaseVelocity(env.robot_id)[0][:2]), 0.03)

    def test_changed_course_is_not_a_fixed_trajectory(self):
        env = self.make_env(layout="course")
        first = env.obstacle_specs.copy()
        env.reset(seed=7)
        self.assertNotEqual(first, env.obstacle_specs)
        info = self.walk_to_goal(env)
        self.assertGreater(info["min_obstacle_clearance"], 0.10)

    def test_new_obstacle_during_walking_is_seen_and_avoided(self):
        env = self.make_env(layout="empty")
        for _ in range(360):
            env.step(np.zeros(12))
        self.assertEqual(env.navigator.seen_hit_count, 0)
        env.add_box_ahead()
        info = self.walk_to_goal(env)
        self.assertGreater(info["observed_cells"], 0)
        self.assertGreater(info["min_obstacle_clearance"], 0.10)

    def test_impassable_wall_stops_without_contact(self):
        env = self.make_env(layout="blocked")
        for _ in range(1800):
            _, _, terminated, _, info = env.step(np.zeros(12))
            self.assertFalse(terminated)
        self.assertFalse(info["goal_reached"])
        self.assertEqual(info["nav_mode"], "STOP_BLOCKED")
        self.assertEqual(info["collision_steps"], 0)
        self.assertLess(info["distance_x"], 1.5)
        self.assertLess(np.linalg.norm(env._p.getBaseVelocity(env.robot_id)[0][:2]), 0.03)

    def test_disabled_avoidance_really_hits_the_obstacle(self):
        env = self.make_env(layout="single", avoidance=False)
        for _ in range(1200):
            _, _, terminated, _, info = env.step(np.zeros(12))
            if terminated:
                break
        self.assertTrue(terminated, "A physical collision must occur when avoidance is disabled.")
        self.assertEqual(info["termination_reason"], "obstacle_collision")
        self.assertGreater(info["collision_steps"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
