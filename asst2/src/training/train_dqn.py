from typing import Dict, Any , List 
from functools import partial

from numpy.random import seed 

import jax 
import jax.numpy as jnp

from src.envs.gridworld import EnvParams
from src.agents.dqn_agent import DQNAgent
from src.networks.q_network import init_q_params, q_forward_batch
from src.replay.replay_buffer import init_buffer, add_transition, sample_batch
from src.evaluate.evaluate_dqn import evaluate_dqn_greedy


def linear_epsilon_decay(
        episode: int,
        epsilon_start: float,
        epsilon_end: float,
        decay_episodes: int
) -> float:
    
    if episode >= decay_episodes:
        return epsilon_end 
    frac = episode / decay_episodes
    return epsilon_start + frac * (epsilon_end - epsilon_start)

def dqn_loss(
        q_params: Dict,
        target_params: Dict,
        batch: Dict[str, jnp.ndarray],
        gamma: float = 0.99
) -> jnp.ndarray:
    """DQN Loss: 
            L = mean (Q(s,a) - y)^2
        where 
            y = r + gamma * max_a' Q_target(s', a') if not done else r
            y = r if done
        """
    q_values = q_forward_batch(q_params, batch["obs"])
    q_sa = jnp.take_along_axis(q_values, batch["actions"][:, None], axis=1).squeeze(axis=1)

    next_q_values = q_forward_batch(target_params, batch["next_obs"])
    max_next_q = jnp.max(next_q_values, axis=1)

    dones = batch["dones"].astype(jnp.float32)
    targets = batch["rewards"] + gamma * max_next_q * (1.0 - dones)

    return jnp.mean((q_sa - targets) ** 2)

@partial(jax.jit, static_argnames=("gamma",))
def update_q_network(
    q_params: Dict,
    target_params: Dict,
    batch: Dict[str, jnp.ndarray],
    learning_rate: float,
    gamma: float = 0.99,
):
    loss_fn = lambda p: dqn_loss(p, target_params, batch, gamma)
    loss, grads = jax.value_and_grad(loss_fn)(q_params)

    new_q_params = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g,
        q_params,
        grads,
    )
    return new_q_params, loss

def train_dqn(
        env, 
        env_params,
        init_q_params: Dict,
        num_episodes: int = 1000,
        max_steps: int = 50,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        seed: int = 0,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        warmup_steps: int = 100,
        target_update_freq: int = 100,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay_episodes: int = 500,
        log_every: int = 50,
        logger: Any = None,
        met_df: Any = None
        ) -> Dict[str, Any]:
    
    key = jax.random.PRNGKey(seed)
    q_params = init_q_params
    target_params = init_q_params

    obs_dim = 2
    buffer = init_buffer(buffer_capacity, obs_dim)

    episode_rewards : List[float] = []
    episode_lengths : List[int] = []
    losses : List[float] = []
    eval_success_rates : List[float] = []

    global_step  = 0 

    for episode in range(1, num_episodes+1):
        epsilon = linear_epsilon_decay(
            episode,
            epsilon_start,
            epsilon_end,
            epsilon_decay_episodes
        )

        agent  = DQNAgent(q_params)
        key, reset_key = jax.random.split(key)
        obs, state = env.reset_env(reset_key, env_params)

        done = False
        ep_reward = 0.0 
        ep_length = 0 
        ep_losses = [] 

        while (not bool(done)) and (ep_length < max_steps):
            key, act_key, step_key, sample_key = jax.random.split(key, 4)

            action = agent.act(act_key, obs, epsilon)

            next_obs, next_state, reward, done, _ = env.step_env(
                step_key, state, action, env_params
            )

            buffer = add_transition(
                buffer=buffer,
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
            )

            obs = next_obs
            state = next_state

            ep_reward += float(reward)
            ep_length += 1
            global_step += 1

            if int(buffer["size"]) >= max(warmup_steps, batch_size):
                batch = sample_batch(buffer, sample_key, batch_size)

                q_params, loss = update_q_network(
                    q_params=q_params,
                    target_params=target_params,
                    batch=batch,
                    learning_rate=learning_rate,
                    gamma=gamma,
                )
                ep_losses.append(float(loss))

                if global_step % target_update_freq == 0:
                    target_params = q_params

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        losses.append(sum(ep_losses) / len(ep_losses) if ep_losses else 0.0)
        logger.info(f"Episode {episode:4d} - Reward: {ep_reward:.3f}, Length: {ep_length}, Loss: {losses[-1]:.4f}, Epsilon: {epsilon:.3f}")
        met_df.add_episode(seed=seed, episode=episode, reward=ep_reward, episode_length=ep_length, loss=losses[-1], algorithm="DQN", lr=learning_rate, gamma=gamma)

        if episode % log_every == 0:
            avg_reward = sum(episode_rewards[-log_every:]) / log_every
            avg_length = sum(episode_lengths[-log_every:]) / log_every
            avg_loss = sum(losses[-log_every:]) / log_every

            eval_stats = evaluate_dqn_greedy(
                env=env,
                env_params=env_params,
                q_params=q_params,
                num_episodes=25,
                max_steps=max_steps,
                seed=seed + episode,
            )
            eval_success_rates.append(eval_stats["success_rate"])

            print(
                f"[Episode {episode:4d}] "
                f"eps={epsilon:.3f}  "
                f"avg_reward={avg_reward:8.3f}  "
                f"avg_length={avg_length:6.2f}  "
                f"avg_loss={avg_loss:8.4f}  "
                f"eval_success={eval_stats['success_rate']:.3f}"
            )
            logger.info(
                f"[Episode {episode:4d}] "
                f"eps={epsilon:.3f}  "
                f"avg_reward={avg_reward:8.3f}  "
                f"avg_length={avg_length:6.2f}  "
                f"avg_loss={avg_loss:8.4f}  "
                f"eval_success={eval_stats['success_rate']:.3f}"
            )

    return {
        "final_q_params": q_params,
        "target_q_params": target_params,
        "episode_rewards": jnp.array(episode_rewards, dtype=jnp.float32),
        "episode_lengths": jnp.array(episode_lengths, dtype=jnp.int32),
        "losses": jnp.array(losses, dtype=jnp.float32),
        "eval_success_rates": jnp.array(eval_success_rates, dtype=jnp.float32),
    }