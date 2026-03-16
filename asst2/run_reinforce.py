"""
run_reinforce.py  —  Standalone REINFORCE runner.
Outputs:
    metrics/reinforce_episodes.csv
    metrics/reinforce_summary.csv
    checkpoints/reinforce_seed=*_gamma=*_lr=*.npz

Usage:
    python run_reinforce.py
    python run_reinforce.py --config path/to/config.yaml
"""

import os
import yaml
import jax
import numpy as np
import jax.numpy as jnp
from typing import Dict

from src.envs.gridworld import ObstacleTrapGridWorld, EnvParams
from src.networks.policy_network import init_policy_params
from src.training.train_reinforce import train_reinforce
from src.evaluate.evaluate_policy import evaluate_policy
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

    logger = setup_logger(run_id=0, path="./logs/reinforce_standalone")
    logger.info("=" * 50)
    logger.info("Starting REINFORCE standalone run")

    # ── Device info ──
    device_info = log_device_info(logger)
    force_jax_gpu_or_warn(logger)
    warmup_jit(logger)

    met_df = RLMetricsDataset(proj_name="reinforce")

    env_cfg = config.get("env", {})
    reinforce_cfg = config["reinforce"]
    seeds = config["seeds"]
    gammas = config["gammas"]
    lrs = config["lrs"]

    env = ObstacleTrapGridWorld()
    env_params = EnvParams(**env_cfg) if env_cfg else EnvParams()

    total_configs = len(seeds) * len(gammas) * len(lrs)
    run_idx = 0

    total_timer = Timer("REINFORCE (full grid)")
    with total_timer:
        for seed in seeds:
            for gamma in gammas:
                for lr in lrs:
                    run_idx += 1
                    tag = f"seed={seed}, gamma={gamma}, lr={lr}"
                    print(f"{'#' * 40}")
                    print(f" [{run_idx}/{total_configs}] REINFORCE  {tag}")
                    logger.info(f"Running REINFORCE  {tag}")

                    key = jax.random.PRNGKey(seed)

                    run_id = int(seed * 100000 + gamma * 100 + lr * 1000)
                    run_logger = setup_logger(run_id, path="./logs/reinforce")

                    params = init_policy_params(
                        key=key,
                        obs_dim=2,
                        hidden_dim=reinforce_cfg["model"]["hidden_dim"],
                        num_actions=4,
                    )

                    timer = Timer(f"REINFORCE {tag}")
                    with timer:
                        results = train_reinforce(
                            env=env,
                            env_params=env_params,
                            init_params=params,
                            num_episodes=reinforce_cfg["num_episodes"],
                            max_steps=reinforce_cfg["max_steps"],
                            learning_rate=lr,
                            gamma=gamma,
                            seed=seed,
                            log_every=reinforce_cfg["log_every"],
                            normalize_returns=True,
                            logger=run_logger,
                            metdf=met_df,
                        )

                    trained_params = results["final_params"]

                    # Save checkpoint
                    save_checkpoint_npz(
                        trained_params,
                        f"checkpoints/reinforce_seed={seed}_gamma={gamma}_lr={lr}.npz",
                    )

                    # ── Evaluate stochastic ──
                    eval_stochastic = evaluate_policy(
                        env, env_params, trained_params,
                        num_episodes=200, seed=seed,
                    )

                    # ── Evaluate greedy ──
                    eval_greedy = evaluate_policy(
                        env, env_params, trained_params,
                        num_episodes=reinforce_cfg["eval_episodes"],
                        greedy=True, seed=seed,
                    )

                    logger.info(
                        f"  Stochastic: reward={eval_stochastic['mean_reward']:.3f} "
                        f"success={eval_stochastic['success_rate']:.3f}"
                    )
                    logger.info(
                        f"  Greedy:     reward={eval_greedy['mean_reward']:.3f} "
                        f"success={eval_greedy['success_rate']:.3f}"
                    )
                    logger.info(timer.report())

                    print(
                        f"  Stochastic: reward={eval_stochastic['mean_reward']:.3f}  "
                        f"success={eval_stochastic['success_rate']:.3f}"
                    )
                    print(
                        f"  Greedy:     reward={eval_greedy['mean_reward']:.3f}  "
                        f"success={eval_greedy['success_rate']:.3f}"
                    )
                    print(f"  {timer.report()}")

                    # ── Log summaries ──
                    met_df.add_summary(
                        seed=seed, algorithm="REINFORCE", lr=lr, gamma=gamma,
                        final_mean_reward=eval_stochastic["mean_reward"],
                        final_success_rate=eval_stochastic["success_rate"],
                        backend=device_info["backend"],
                        devices=str(device_info["devices"]),
                        action="stochastic",
                        mean_length=eval_stochastic["mean_length"],
                        wall_time_s=timer.elapsed,
                    )
                    met_df.add_summary(
                        seed=seed, algorithm="REINFORCE", lr=lr, gamma=gamma,
                        final_mean_reward=eval_greedy["mean_reward"],
                        final_success_rate=eval_greedy["success_rate"],
                        backend=device_info["backend"],
                        devices=str(device_info["devices"]),
                        action="greedy",
                        mean_length=eval_greedy["mean_length"],
                        wall_time_s=timer.elapsed,
                    )

    logger.info(total_timer.report())
    print(total_timer.report())

    # ── Save CSVs ──
    paths = met_df.save(output_dir="metrics", filename="reinforce_episodes.csv")
    logger.info(f"Saved: {paths}")
    print(f"Saved: {paths}")
    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
