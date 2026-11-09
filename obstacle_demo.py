"""Forward walking and whole-body turning during autonomous obstacle avoidance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pybullet as p

from obstacle_env import ObstacleAvoidanceEnv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=None,
                        help="Duration cap; headless default 120 s, GUI default continuous.")
    parser.add_argument("--layout", choices=("course", "single", "blocked", "empty"), default="course")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-rays", action="store_true")
    parser.add_argument("--no-avoidance", action="store_true", help="Collision control experiment only.")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be positive and finite.")
    duration = args.seconds if args.seconds is not None else (120.0 if args.headless else None)
    limit = None if duration is None else max(1, math.ceil(duration * 60))
    env = ObstacleAvoidanceEnv(render=not args.headless, layout=args.layout,
                               avoidance=not args.no_avoidance,
                               max_steps=max(7200, limit or 2_147_483_647))
    report = {"nav_version": env.NAV_VERSION, "layout": args.layout,
              "seed": args.seed, "avoidance_enabled": env.avoidance,
              "status": "running", "resets": 0, "collisions": 0, "falls": 0}
    current_seed = args.seed
    info = None
    steps = 0
    trace = []
    max_roll = max_pitch = 0.0
    goal_at = None
    blocked_at = None
    code = 0
    try:
        _, info = env.reset(seed=current_seed)
        print("HEADING AVOIDANCE | SPACE: pause | R: restart | N: new course | O: box in front of head | L: laser | ESC: exit", flush=True)
        paused = False
        show_rays = args.show_rays
        deadline = time.perf_counter()
        while limit is None or steps < limit:
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
                if keys.get(ord("l"), 0) & p.KEY_WAS_TRIGGERED:
                    show_rays = not show_rays
                reset = keys.get(ord("r"), 0) & p.KEY_WAS_TRIGGERED
                new = keys.get(ord("n"), 0) & p.KEY_WAS_TRIGGERED
                if reset or new:
                    current_seed += int(bool(new))
                    _, info = env.reset(seed=current_seed)
                    report["resets"] += 1
                    steps = 0
                    trace = []
                    max_roll = max_pitch = 0.0
                    goal_at = blocked_at = None
                    deadline = time.perf_counter()
                if keys.get(ord("o"), 0) & p.KEY_WAS_TRIGGERED:
                    if not info["goal_reached"]:
                        env.add_box_ahead()
                        print("Added a collidable box ahead. The next laser scan will replan.", flush=True)
                if paused:
                    env.update_navigation_display(show_rays=show_rays)
                    time.sleep(0.02)
                    deadline = time.perf_counter()
                    continue
            _, _, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
            steps += 1
            position = env._base_pos()[:2]
            max_roll = max(max_roll, abs(info["roll"]))
            max_pitch = max(max_pitch, abs(info["pitch"]))
            if steps % 6 == 0:
                trace.append({"t": env.elapsed_time, "x": float(position[0]), "y": float(position[1]),
                              "mode": info["nav_mode"], "front_m": info["front_distance"],
                              "yaw": info["yaw"], "desired_heading": info["desired_heading"]})
            if steps % 120 == 0:
                print(f"t={env.elapsed_time:6.1f}s | {info['nav_mode']:12s} | "
                      f"xy=({position[0]:+.2f},{position[1]:+.2f})m | "
                      f"heading={math.degrees(info['yaw']):+.1f}deg | "
                      f"front={info['front_distance']:.2f}m | goal={info['goal_distance']:.2f}m", flush=True)
            if terminated:
                if info.get("termination_reason") == "obstacle_collision":
                    report["status"] = "collision"
                    report["collisions"] += 1
                else:
                    report["status"] = "fell"
                    report["falls"] += 1
                code = 1
                break
            if info["goal_reached"] and goal_at is None:
                goal_at = env.elapsed_time
                print("Goal reached. Stopping the feet and holding position.", flush=True)
            if info["nav_mode"] == "STOP_BLOCKED":
                if blocked_at is None:
                    blocked_at = env.elapsed_time
            else:
                blocked_at = None
            if args.headless and goal_at is not None and env.elapsed_time - goal_at >= 3.0:
                report["status"] = "goal_reached"
                break
            if args.headless and blocked_at is not None and env.elapsed_time - blocked_at >= 8.0:
                report["status"] = "stopped_blocked"
                break
            if truncated:
                report["status"] = "time_limit"
                break
            if not args.headless:
                if steps % 12 == 0:
                    env.update_navigation_display(show_rays=show_rays)
                env.render()
                deadline += env.control_dt
                delay = deadline - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.25:
                    deadline = time.perf_counter()
        if report["status"] == "running":
            report["status"] = "goal_reached" if info and info["goal_reached"] else "time_limit"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    except Exception as exc:
        if isinstance(exc, p.error) and not env.is_connected:
            report["status"] = "window_closed"
        else:
            report.update(status="error", error=str(exc))
            print(f"ERROR: {exc}", file=sys.stderr)
            code = 1
    finally:
        if info is not None and env.is_connected:
            position = env._base_pos()[:2]
            passed = sum(position[0] > spec["xy"][0] + spec["size"][0] / 2 + 0.4
                         for spec in env.obstacle_specs)
            # Average movement over a full gait cycle, using torso yaw at the
            # interval midpoint, to measure heading versus actual travel.
            alignment_errors = []
            for index in range(5, len(trace)):
                dx = trace[index]["x"] - trace[index - 5]["x"]
                dy = trace[index]["y"] - trace[index - 5]["y"]
                if math.hypot(dx, dy) > 0.04:
                    difference = env.wrap_angle(trace[index - 2]["yaw"] - math.atan2(dy, dx))
                    alignment_errors.append(abs(math.degrees(difference)))
            headings = [math.degrees(row["yaw"]) for row in trace]
            report.update(
                seed=current_seed, steps=steps, simulated_seconds=env.elapsed_time,
                final_xy=position.tolist(), goal=env.goal.tolist(),
                goal_reached=info["goal_reached"], goal_distance=info["goal_distance"],
                nav_mode=info["nav_mode"], min_obstacle_clearance=info["min_obstacle_clearance"],
                max_abs_roll_degrees=math.degrees(max_roll), max_abs_pitch_degrees=math.degrees(max_pitch),
                replans=info["replans"], observed_cells=info["observed_cells"],
                obstacles_passed=int(passed), obstacle_count=len(env.obstacle_ids),
                obstacles=env.obstacle_specs, goal_reached_at_seconds=goal_at, trace=trace,
                min_heading_degrees=min(headings, default=0.0),
                max_heading_degrees=max(headings, default=0.0),
                mean_body_travel_misalignment_degrees=float(np.mean(alignment_errors)) if alignment_errors else None,
                p95_body_travel_misalignment_degrees=float(np.percentile(alignment_errors, 95)) if alignment_errors else None,
            )
        env.close()
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = {key: value for key, value in report.items() if key not in ("trace", "obstacles")}
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
