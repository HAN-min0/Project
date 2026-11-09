"""Physics regression tests. Run: python test_walk.py (about a minute)."""

import math
import unittest

import numpy as np
import pybullet as p

from go2_env import Go2GymEnv


class WalkingTests(unittest.TestCase):
    def test_default_walk_sixty_seconds(self):
        env = Go2GymEnv(max_steps=3600)
        self.addCleanup(env.close)
        observation, _ = env.reset(seed=42)
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(observation))
        lifted = np.zeros(4, dtype=bool)
        touched = np.zeros(4, dtype=bool)
        max_lateral = 0.0
        for step in range(3600):
            observation, reward, terminated, truncated, info = env.step(np.zeros(12))
            self.assertFalse(terminated, f"Fell at step {step}: {info}")
            self.assertEqual(truncated, step == 3599)
            self.assertTrue(env.observation_space.contains(observation))
            self.assertTrue(math.isfinite(reward))
            max_lateral = max(max_lateral, abs(info["lateral_drift"]))
            if step > 120:
                contact = np.asarray(info["foot_contacts"])
                touched |= contact
                lifted |= ~contact
        self.assertGreater(info["distance_x"], 10.0)
        self.assertLess(max_lateral, 0.3)
        self.assertGreater(info["height"], 0.3)
        self.assertTrue(touched.all(), "Each toe must support the robot.")
        self.assertTrue(lifted.all(), "Each toe must leave the ground during walking.")

    def test_standing_does_not_drift(self):
        env = Go2GymEnv(target_speed=0.0, max_steps=600)
        self.addCleanup(env.close)
        env.reset(seed=1)
        for _ in range(600):
            _, _, terminated, _, info = env.step(np.zeros(12))
            self.assertFalse(terminated)
        self.assertLess(abs(info["distance_x"]), 0.05)
        self.assertLess(abs(info["lateral_drift"]), 0.05)

    def test_reset_and_time_limit(self):
        env = Go2GymEnv(max_steps=3)
        self.addCleanup(env.close)
        first, _ = env.reset(seed=123)
        for step in range(3):
            obs, _, terminated, truncated, _ = env.step(np.zeros(12))
            self.assertFalse(terminated)
            self.assertEqual(truncated, step == 2)
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(12))
        reset, _ = env.reset(seed=123)
        np.testing.assert_array_equal(first, reset)
        self.assertEqual(env.step_counter, 0)

    def test_fall_is_reported_without_automatic_reset(self):
        env = Go2GymEnv()
        self.addCleanup(env.close)
        env.reset()
        env._p.resetBasePositionAndOrientation(
            env.robot_id, [0, 0, 0.48], p.getQuaternionFromEuler([1.1, 0, 0]),
        )
        _, _, terminated, _, info = env.step(np.zeros(12))
        self.assertTrue(terminated)
        self.assertEqual(info["termination_reason"], "fall_or_invalid_state")
        self.assertEqual(env.step_counter, 1)

    def test_invalid_actions_do_not_advance_physics(self):
        env = Go2GymEnv()
        self.addCleanup(env.close)
        env.reset()
        for invalid in (np.zeros(11), np.zeros((1, 12)), np.full(12, np.nan), np.full(12, np.inf)):
            with self.assertRaises(ValueError):
                env.step(invalid)
        self.assertEqual(env.step_counter, 0)

    def test_clients_are_isolated_and_close_is_repeatable(self):
        first, second = Go2GymEnv(), Go2GymEnv()
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        first.reset()
        expected, _ = second.reset()
        first.step(np.zeros(12))
        np.testing.assert_array_equal(second._get_obs(), expected)
        first.close()
        first.close()
        self.assertTrue(second.is_connected)
        second.step(np.zeros(12))

    def test_sb3_environment_contract(self):
        try:
            from stable_baselines3.common.env_checker import check_env
        except ImportError:
            self.skipTest("Optional stable-baselines3 package is not installed.")
        env = Go2GymEnv()
        self.addCleanup(env.close)
        check_env(env, warn=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
