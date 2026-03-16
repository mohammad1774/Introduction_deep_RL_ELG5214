"""
run_dqn.py  —  Standalone DQN runner.
Outputs:
    metrics/dqn_episodes.csv
    metrics/dqn_summary.csv
    checkpoints/dqn_seed=*_gamma=*_lr=*.npz

Usage:
    python run_dqn.py
    python run_dqn.py --config path/to/config.yaml
"""

import os
import yaml
import jax
import numpy as np
import jax.numpy as jnp
from typing import Dict

from src.envs.gridworld import ObstacleTrapGridWorld, EnvParams
from src.networks.q_network import init_q_params
from src.training.train_dqn import train_dqn
from src.evaluate.evaluate_dqn import evaluate_dqn_greedy
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


def save_checkpoint_npz(params: Dict[str, jnp.ndarray], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    params_np = {k: np.array(v) for k, v in params.items()}
    np.savez(path, **params_np)


def main(config_path: str = "config.yaml"):
    config = load_config(config_path)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    logger = setup_logger(run_id=0, path="./logs/dqn_standalone")
    logger.info("=" * 50)
    logger.info("Starting DQN standalone run")

    # ── Device info ──
    device_info = log_device_info(logger)
    force_jax_gpu_or_warn(logger)
    warmup_jit(logger)

    met_df = RLMetricsDataset(proj_name="dqn")

    env_cfg = config.get("env", {})
    dqn_cfg = config["dqn"]
    seeds = config["seeds"]
    gammas = config["gammas"]
    lrs = config["lrs"]

    env = ObstacleTrapGridWorld()
    env_params = EnvParams(**env_cfg) if env_cfg else EnvParams()

    total_configs = len(seeds) * len(gammas) * len(lrs)
    run_idx = 0

    total_timer = Timer("DQN (full grid)")
    with total_timer:
        for seed in seeds:
            for gamma in gammas:
                for lr in lrs:
                    run_idx += 1
                    tag = f"seed={seed}, gamma={gamma}, lr={lr}"
                    print(f"{'#' * 40}")
                    print(f" [{run_idx}/{total_configs}] DQN  {tag}")
                    logger.info(f"Running DQN  {tag}")

                    key = jax.random.PRNGKey(seed)

                    init_params = init_q_params(
                        key=key,
                        obs_dim=2,
                        hidden_dim=dqn_cfg["model"]["hidden_dim"],
                        num_actions=4,
                    )

                    run_id = int(seed * 100000 + gamma * 100 + lr * 1000)
                    run_logger = setup_logger(run_id, path="./logs/dqn")

                    timer = Timer(f"DQN {tag}")
                    with timer:
                        results = train_dqn(
                            env=env,
                            env_params=env_params,
                            init_q_params=init_params,
                            num_episodes=dqn_cfg["num_episodes"],
                            max_steps=dqn_cfg["max_steps"],
                            learning_rate=lr,
                            gamma=gamma,
                            seed=seed,
                            buffer_capacity=dqn_cfg["buffer_capacity"],
                            batch_size=dqn_cfg["batch_size"],
                            warmup_steps=dqn_cfg["warmup_steps"],
                            target_update_freq=dqn_cfg["target_update_freq"],
                            epsilon_start=dqn_cfg["epsilon_start"],
                            epsilon_end=dqn_cfg["epsilon_end"],
                            epsilon_decay_episodes=dqn_cfg["epsilon_decay_episodes"],
                            log_every=dqn_cfg["log_every"],
                            logger=run_logger,
                            met_df=met_df,
                        )

                    # Save checkpoint
                    save_checkpoint_npz(
                        results["final_q_params"],
                        f"checkpoints/dqn_seed={seed}_gamma={gamma}_lr={lr}.npz",
                    )

                    # ── Evaluate greedy ──
                    eval_stats = evaluate_dqn_greedy(
                        env=env,
                        env_params=env_params,
                        q_params=results["final_q_params"],
                        num_episodes=100,
                        max_steps=50,
                        seed=123,
                    )

                    logger.info(
                        f"  Greedy eval: reward={eval_stats['mean_reward']:.3f} "
                        f"success={eval_stats['success_rate']:.3f}"
                    )
                    logger.info(timer.report())

                    print(
                        f"  Greedy eval: reward={eval_stats['mean_reward']:.3f}  "
                        f"success={eval_stats['success_rate']:.3f}"
                    )
                    print(f"  {timer.report()}")

                    # ── Log summary ──
                    met_df.add_summary(
                        seed=seed, algorithm="DQN", lr=lr, gamma=gamma,
                        final_mean_reward=eval_stats["mean_reward"],
                        final_success_rate=eval_stats["success_rate"],
                        backend=device_info["backend"],
                        devices=str(device_info["devices"]),
                        action="greedy",
                        mean_length=eval_stats["mean_length"],
                        wall_time_s=timer.elapsed,
                    )

    logger.info(total_timer.report())
    print(total_timer.report())

    # ── Save CSVs ──
    paths = met_df.save(output_dir="metrics", filename="dqn_episodes.csv")
    logger.info(f"Saved: {paths}")
    print(f"Saved: {paths}")
    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
