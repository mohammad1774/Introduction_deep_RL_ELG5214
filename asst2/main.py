import os
import yaml
import jax

from src.utils.reusable import (
    RLMetricsDataset,
    setup_logger,
    log_device_info,
    force_jax_gpu_or_warn,
    warmup_jit,
    Timer,
)
from src.test.test_random_agent import test_random_agent
from src.test.test_reinforce_agent import test_reinforce_agent
from src.test.test_dqn_agent import test_dqn_agent
from typing import Dict, Any
import jax.numpy as jnp
import numpy as np


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint_npz(params: Dict[str, jnp.ndarray], path: str):
    """Saves JAX arrays as .npz (portable). This is torch.save equivalent."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    params_np = {k: np.array(v) for k, v in params.items()}
    np.savez(path, **params_np)


def main(config_path: str = "config.yaml"):
    config = load_config(config_path)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    logger = setup_logger(run_id=0, path="./logs/main")
    logger.info("Starting Assignment 2 main run")

    # ─── GPU / Device detection ───
    device_info = log_device_info(logger)
    backend = force_jax_gpu_or_warn(logger)
    warmup_jit(logger)

    # Shared metrics object for all algorithms
    met_df = RLMetricsDataset(proj_name="assignment2_rl")

    env_cfg = config.get("env", {})

    # ─────────────────────────────
    #  Random Agent  (seed only — no gamma/lr needed)
    # ─────────────────────────────
    if config.get("run_random", True):
        random_cfg = config["random"]
        for seed in config["seeds"]:
            print("#" * 40)
            print(f" Random Agent  seed={seed}")
            logger.info(f"Running Random Agent baseline  seed={seed}")

            timer = Timer(f"RandomAgent seed={seed}")
            with timer:
                test_random_agent(
                    seed=seed,
                    num_episodes=random_cfg["num_episodes"],
                    max_steps=random_cfg["max_steps"],
                    config_env_params=env_cfg,
                    met_df=met_df,
                )
            logger.info(timer.report())
            print(timer.report())

    # ─────────────────────────────
    #  REINFORCE + DQN  (full hp grid)
    # ─────────────────────────────
    total_timer = Timer("Full HP grid (REINFORCE + DQN)")
    with total_timer:
        for seed in config["seeds"]:
            for gamma in config["gammas"]:
                for lr in config["lrs"]:
                    tag = f"seed={seed}, gamma={gamma}, lr={lr}"
                    logger.info(f"Running experiments for {tag}")

                    # ── REINFORCE ──
                    if config.get("run_reinforce", True):
                        print("#" * 40)
                        print(f" REINFORCE  {tag}")
                        logger.info(f"Running REINFORCE  {tag}")
                        reinforce_cfg = config["reinforce"]

                        timer = Timer(f"REINFORCE {tag}")
                        with timer:
                            reinforce_params, reinforce_eval_stats, reinforce_eval_stats_greedy = (
                                test_reinforce_agent(
                                    seed=seed,
                                    gamma=gamma,
                                    lr=lr,
                                    config=reinforce_cfg,
                                    config_env_params=env_cfg,
                                    met_df=met_df,
                                )
                            )
                        save_checkpoint_npz(
                            reinforce_params,
                            f"checkpoints/reinforce_seed={seed}_gamma={gamma}_lr={lr}.npz",
                        )
                        logger.info(timer.report())
                        print(timer.report())

                    # ── DQN ──
                    if config.get("run_dqn", True):
                        print("#" * 40)
                        print(f" DQN  {tag}")
                        logger.info(f"Running DQN  {tag}")
                        dqn_cfg = config["dqn"]

                        timer = Timer(f"DQN {tag}")
                        with timer:
                            dqn_params, dqn_eval_stats = test_dqn_agent(
                                seed=seed,
                                gamma=gamma,
                                lr=lr,
                                config_env_params=env_cfg,
                                config=dqn_cfg,
                                met_df=met_df,
                                logger=logger,
                            )
                        save_checkpoint_npz(
                            dqn_params,
                            f"checkpoints/dqn_seed={seed}_gamma={gamma}_lr={lr}.npz",
                        )
                        logger.info(timer.report())
                        print(timer.report())

    logger.info(total_timer.report())
    print(total_timer.report())

    # ─────────────────────────────
    #  Save all metrics together
    # ─────────────────────────────
    save_paths = met_df.save(
        output_dir="metrics", filename="assignment2_all_algorithms.csv"
    )

    logger.info(f"Saved metrics: {save_paths}")
    logger.info(f"Compute backend used: {device_info['backend']}")
    print("Saved metrics:", save_paths)
    print(f"Compute backend: {device_info['backend']}")
    print("Main run finished.")


if __name__ == "__main__":
    main()
