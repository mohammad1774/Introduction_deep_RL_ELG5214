"""
run_random.py  —  Standalone Random Agent runner.
Outputs:
    metrics/random_episodes.csv
    metrics/random_summary.csv

Usage:
    python run_random.py
    python run_random.py --config path/to/config.yaml
"""

import os
import yaml
import jax

from src.envs.gridworld import ObstacleTrapGridWorld, EnvParams
from src.agents.random_agent import RandomAgent
from src.evaluate.evaluate_random import evaluate_random_agent
from src.utils.reusable import (
    RLMetricsDataset,
    setup_logger,
    log_device_info,
    force_jax_gpu_or_warn,
    warmup_jit,
    Timer,
)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str = "config.yaml"):
    config = load_config(config_path)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    logger = setup_logger(run_id=0, path="./logs/random_agent")
    logger.info("=" * 50)
    logger.info("Starting Random Agent standalone run")

    # ── Device info ──
    device_info = log_device_info(logger)
    force_jax_gpu_or_warn(logger)
    warmup_jit(logger)

    met_df = RLMetricsDataset(proj_name="random")

    env_cfg = config.get("env", {})
    random_cfg = config["random"]
    seeds = config["seeds"]

    env = ObstacleTrapGridWorld()
    env_params = EnvParams(**env_cfg) if env_cfg else EnvParams()

    total_timer = Timer("Random Agent (all seeds)")
    with total_timer:
        for seed in seeds:
            tag = f"seed={seed}"
            print(f"{'#' * 40}")
            print(f" Random Agent  {tag}")
            logger.info(f"Running Random Agent  {tag}")

            agent = RandomAgent()

            timer = Timer(f"RandomAgent {tag}")
            with timer:
                stats = evaluate_random_agent(
                    env, env_params, agent,
                    num_episodes=random_cfg["num_episodes"],
                    max_steps=random_cfg["max_steps"],
                    seed=seed,
                )

            logger.info(f"Stats: {stats}")
            logger.info(timer.report())
            print(f"  avg_reward={stats['average_reward']:.3f}  "
                  f"success={stats['success_rate']:.3f}  "
                  f"| {timer.report()}")

            met_df.add_summary(
                seed=seed,
                algorithm="RandomAgent",
                lr=0.0,
                gamma=0.0,
                final_mean_reward=stats["average_reward"],
                final_success_rate=stats["success_rate"],
                backend=device_info["backend"],
                devices=str(device_info["devices"]),
                action="random",
                mean_length=stats["average_length"],
                wall_time_s=timer.elapsed,
            )

    logger.info(total_timer.report())
    print(total_timer.report())

    # ── Save CSVs ──
    paths = met_df.save(output_dir="metrics", filename="random_episodes.csv")
    logger.info(f"Saved: {paths}")
    print(f"Saved: {paths}")
    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
