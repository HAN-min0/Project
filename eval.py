"""Run the walking demo. Default controller needs no trained model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pybullet as p

from go2_env import Go2GymEnv


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "go2_walk.zip"


def load_policy(path, env):
    """Check a policy's environment contract before loading it."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found: {path}\nRun go2_train.py, or use --controller gait."
        )
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        raise ValueError("Missing model JSON metadata. Old 32-element policies are incompatible; retrain with go2_train.py.")
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if data.get("env_version") != env.ENV_VERSION or not math.isclose(
        data.get("target_speed", -1), env.target_speed, abs_tol=1e-9
    ):
        raise ValueError("Model environment version or target speed does not match this demo.")
    from stable_baselines3 import PPO

    policy = PPO.load(path, env=env, device="cpu")
    return policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run physics without a window.")
    parser.add_argument("--seconds", type=float, default=None, help="Simulation duration; GUI default is continuous, headless default is 30 s.")
    parser.add_argument("--controller", choices=("gait", "ppo"), default="gait")
    parser.add_argument("--speed", type=float, choices=(0.25, 0.30), default=0.25, help="Tested forward speed in m/s.")
    parser.add_argument("--stand", action="store_true", help="Stand without walking, for comparison.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path, help="Write a JSON result to this path.")
    args = parser.parse_args(argv)
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be positive and finite.")
    duration = args.seconds if args.seconds is not None else (30.0 if args.headless else None)
    limit = None if duration is None else max(1, math.ceil(duration / Go2GymEnv.CONTROL_DT))
    env = Go2GymEnv(render=not args.headless, target_speed=0.0 if args.stand else args.speed,
                    max_steps=max(3600, limit or 2_147_483_647))
    report = {
        "controller": args.controller, "env_version": env.ENV_VERSION,
        "target_speed": env.target_speed, "requested_seconds": duration,
        "status": "running", "steps": 0, "falls": 0,
    }
    exit_code = 0
    total_steps = 0
    completed_distance = 0.0
    max_roll = max_pitch = max_lateral = 0.0
    restarts = 0
    info = None
    try:
        policy = load_policy(args.model, env) if args.controller == "ppo" else None
        observation, info = env.reset(seed=args.seed)
        print("Walking simulation. SPACE: pause/resume | R: reset | ESC / close window: exit", flush=True)
        paused = False
        deadline = time.perf_counter()
        while limit is None or total_steps < limit:
            if not env.is_connected:
                report["status"] = "window_closed"
                break
            if not args.headless:
                keys = env.keyboard_events()
                if keys.get(27, 0) & p.KEY_WAS_TRIGGERED:
                    report["status"] = "user_exit"
                    break
                if keys.get(ord(" "), 0) & p.KEY_WAS_TRIGGERED:
                    paused = not paused
                if keys.get(ord("r"), 0) & p.KEY_WAS_TRIGGERED:
                    completed_distance += info["distance_x"]
                    observation, info = env.reset(seed=args.seed)
                    restarts += 1
                    deadline = time.perf_counter()
                if paused:
                    time.sleep(0.02)
                    deadline = time.perf_counter()
                    continue
            action = np.zeros(12, dtype=np.float32) if policy is None else policy.predict(observation, deterministic=True)[0]
            observation, _, terminated, truncated, info = env.step(action)
            total_steps += 1
            max_roll = max(max_roll, abs(info["roll"]))
            max_pitch = max(max_pitch, abs(info["pitch"]))
            max_lateral = max(max_lateral, abs(info["lateral_drift"]))
            if total_steps % 60 == 0:
                print(f"t={total_steps * env.control_dt:6.1f}s  x={info['distance_x']:7.3f}m  "
                      f"vx={info['forward_velocity']:+.3f}m/s  "
                      f"roll={math.degrees(info['roll']):+.1f}deg  pitch={math.degrees(info['pitch']):+.1f}deg", flush=True)
            if terminated:
                # A fall is a failed run, not a hidden automatic restart.
                report["falls"] += 1
                report["status"] = "fell"
                exit_code = 1
                print("Robot fell. This run has stopped; its failure is included in the report.", file=sys.stderr)
                break
            if truncated:
                if limit is not None and total_steps >= limit:
                    report["status"] = "completed"
                    break
                completed_distance += info["distance_x"]
                observation, info = env.reset(seed=args.seed)
                restarts += 1
            if not args.headless:
                env.render()
                deadline += env.control_dt
                delay = deadline - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.25:
                    deadline = time.perf_counter()
        if report["status"] == "running":
            report["status"] = "completed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    except Exception as exc:
        # Closing the GUI can race with a physics call on Windows.
        if isinstance(exc, p.error) and not env.is_connected:
            report["status"] = "window_closed"
        else:
            report["status"] = "error"
            report["error"] = str(exc)
            print(f"ERROR: {exc}", file=sys.stderr)
            exit_code = 1
    finally:
        if info is not None:
            report.update(
                steps=total_steps, simulated_seconds=total_steps * env.control_dt,
                distance_x=completed_distance + info["distance_x"],
                final_height=info["height"], final_lateral_drift=info["lateral_drift"],
                max_abs_roll_degrees=math.degrees(max_roll),
                max_abs_pitch_degrees=math.degrees(max_pitch),
                max_abs_lateral_drift=max_lateral, resets=restarts,
            )
        env.close()
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
