"""Optional PPO training: learn small corrections around the walking gait."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
from datetime import datetime, timezone

from go2_env import Go2GymEnv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=float, choices=(0.25, 0.30), default=0.25)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "models" / "go2_walk.zip")
    args = parser.parse_args(argv)
    if args.timesteps < 1:
        parser.error("--timesteps must be positive.")
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.logger import configure
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        parser.exit(1, f"Missing training package: {exc}\nInstall: python -m pip install -r requirements-rl.txt\n")

    # Small MLP policies usually benefit from CPU execution and limited threads.
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    output = args.output.resolve().with_suffix(".zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    env = Go2GymEnv(target_speed=args.speed, max_steps=1800)
    try:
        check_env(env, warn=True)
        monitored = Monitor(env, filename=str(output.parent / "training.monitor.csv"))
        model = PPO(
            "MlpPolicy", monitored, n_steps=1024, batch_size=128,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
            ent_coef=0.005, verbose=1, seed=args.seed, device="cpu",
        )
        model.set_logger(configure(str(output.parent / "ppo_logs"), ["stdout", "csv"]))
        status = "completed"
        try:
            model.learn(total_timesteps=args.timesteps)
        except KeyboardInterrupt:
            status = "interrupted"
            print("Training interrupted. Saving the current policy; evaluate it before use.")
        model.save(output)
        metadata = {
            "env_version": env.ENV_VERSION,
            "observation_size": env.observation_space.shape[0],
            "action_size": env.action_space.shape[0],
            "target_speed": env.target_speed,
            "residual_scale": env.residual_scale,
            "seed": args.seed, "num_timesteps": int(model.num_timesteps),
            "training_status": status,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_required": True,
            "packages": {name: importlib.metadata.version(name) for name in
                         ("numpy", "pybullet", "gymnasium", "stable-baselines3", "torch")},
        }
        output.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(f"Saved policy: {output}")
        print("Evaluate with eval.py --controller ppo --headless --seconds 60 before relying on it.")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
