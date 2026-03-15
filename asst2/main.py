import os
import yaml
import jax

from src.utils.reusable import RLMetricsDataset, setup_logger
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
    """
    Saves JAX arrays as .npz (portable). This is torch.save equivalent.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    params_np = {k: np.array(v) for k, v in params.items()}
    np.savez(path, **params_np)

def main(config_path: str = "config.yaml"):
    config = load_config(config_path)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    logger = setup_logger(run_id=0, path="./logs/main")
    logger.info("Starting Assignment 2 main run")
    logger.info(f"JAX devices: {jax.devices()}")

    # Shared metrics object for all algorithms
    met_df = RLMetricsDataset(proj_name="assignment2_rl")

    env_cfg = config.get("env", {})


    # -------------------------
    # Random Agent
    # -------------------------

    for seed in config["seeds"]:
         for gamma in config["gammas"]:
             for lr in config["lrs"]:
                logger.info(f"Running experiments for seed={seed}, gamma={gamma}, lr={lr}")
    
                    # -------------------------
                    # RANDOM AGENT
                    # -------------------------

                print(f"#" * 20 ) 
                print(f" Random Agent with seed={seed}, gamma={gamma}, lr={lr}")
                if config.get("run_random", True):
                    logger.info("Running Random Agent baseline")
                    random_cfg = config["random"]

                    test_random_agent(
                        seed=seed,
                        num_episodes=random_cfg["num_episodes"],
                        max_steps=random_cfg["max_steps"],
                        config_env_params=env_cfg,
                        met_df=met_df,
                    )
                print(f"#" * 20 )

                # -------------------------
                # REINFORCE
                # -------------------------
                print(f" REINFORCE Agent with seed={seed}, gamma={gamma}, lr={lr}")
                if config.get("run_reinforce", True):
                    logger.info("Running REINFORCE")
                    reinforce_cfg = config["reinforce"]

                    reinforce_params, reinforce_eval_stats , reinforce_eval_stats_greedy = test_reinforce_agent(
                        seed=seed,
                        gamma=gamma,
                        lr=lr,
                        config=reinforce_cfg,
                        config_env_params=env_cfg,
                        met_df=met_df,
                    )
                    save_checkpoint_npz(reinforce_params, f"checkpoints/reinforce_seed={seed}_gamma={gamma}_lr={lr}.npz")

                # -------------------------
                # DQN
                # -------------------------
                print("#" * 20 )
                print(f" DQN Agent with seed={seed}, gamma={gamma}, lr={lr}")
                #print(f" DQN Agent with seed={seed}, gamma={gamma}, lr={lr}")
                if config.get("run_dqn", True):
                    logger.info("Running DQN")
                    dqn_cfg = config["dqn"]

                    dqn_params, dqn_eval_stats = test_dqn_agent(
                        seed=seed,
                        gamma = gamma,
                        lr = lr,
                        config_env_params=env_cfg,
                        config=dqn_cfg,
                        met_df=met_df,
                        logger = logger
                    )
                    save_checkpoint_npz(dqn_params, f"checkpoints/dqn_seed={seed}_gamma={gamma}_lr={lr}.npz")

    # -------------------------
    # Save all metrics together
    # -------------------------
    save_paths = met_df.save(output_dir="metrics", filename="assignment2_all_algorithms.csv")

    logger.info(f"Saved metrics: {save_paths}")
    print("Saved metrics:")
    print(save_paths)
    print("Main run finished.")


if __name__ == "__main__":
    main()
