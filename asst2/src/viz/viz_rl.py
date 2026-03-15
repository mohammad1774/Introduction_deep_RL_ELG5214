import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def standard_error(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) <= 1:
        return 0.0
    return x.std(ddof=1) / math.sqrt(len(x))


def smooth_series(y, window=10):
    if window <= 1:
        return np.asarray(y)
    return pd.Series(y).rolling(window=window, min_periods=1).mean().to_numpy()


def load_metrics(episode_csv: str, summary_csv: str):
    ep_df = pd.read_csv(episode_csv)
    sum_df = pd.read_csv(summary_csv)
    return ep_df, sum_df


def plot_mean_se_by_algorithm(
    ep_df: pd.DataFrame,
    out_path: str,
    reward_col: str = "reward",
    episode_col: str = "episode",
    algorithm_col: str = "algorithm",
    seed_col: str = "seed",
    smooth_window: int = 1,
):
    plt.figure(figsize=(10, 6))

    for algo in sorted(ep_df[algorithm_col].unique()):
        algo_df = ep_df[ep_df[algorithm_col] == algo].copy()

        stats = (
            algo_df.groupby(episode_col)[reward_col]
            .agg(["mean", standard_error])
            .reset_index()
            .rename(columns={"standard_error": "se"})
        )

        x = stats[episode_col].to_numpy()
        y = smooth_series(stats["mean"].to_numpy(), window=smooth_window)
        se = smooth_series(stats["se"].to_numpy(), window=smooth_window)

        plt.plot(x, y, label=algo)
        plt.fill_between(x, y - se, y + se, alpha=0.2)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Mean ± Standard Error of Reward vs Episode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_individual_seed_overlays(
    ep_df: pd.DataFrame,
    out_dir: str,
    reward_col: str = "reward",
    episode_col: str = "episode",
    algorithm_col: str = "algorithm",
    seed_col: str = "seed",
    smooth_window: int = 1,
):
    ensure_dir(out_dir)

    for algo in sorted(ep_df[algorithm_col].unique()):
        plt.figure(figsize=(10, 6))
        algo_df = ep_df[ep_df[algorithm_col] == algo].copy()

        for seed in sorted(algo_df[seed_col].unique()):
            seed_df = algo_df[algo_df[seed_col] == seed].sort_values(episode_col)
            x = seed_df[episode_col].to_numpy()
            y = smooth_series(seed_df[reward_col].to_numpy(), window=smooth_window)
            plt.plot(x, y, alpha=0.7, label=f"seed={seed}")

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Individual Seed Reward Curves: {algo}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{algo}_seed_overlay.png"), dpi=200)
        plt.close()


def plot_hyperparameter_sensitivity(
    sum_df: pd.DataFrame,
    out_path: str,
    algorithm_col: str = "algorithm",
    lr_col: str = "learning_rate",
    gamma_col: str = "gamma",
    score_col: str = "final_mean_reward",
):
    # Mean score per (algorithm, lr, gamma)
    agg = (
        sum_df.groupby([algorithm_col, lr_col, gamma_col])[score_col]
        .mean()
        .reset_index()
    )

    algorithms = sorted(agg[algorithm_col].unique())

    fig, axes = plt.subplots(len(algorithms), 1, figsize=(10, 4 * len(algorithms)))
    if len(algorithms) == 1:
        axes = [axes]

    for ax, algo in zip(axes, algorithms):
        sub = agg[agg[algorithm_col] == algo].copy()

        pivot = sub.pivot(index=gamma_col, columns=lr_col, values=score_col)
        im = ax.imshow(pivot.values, aspect="auto")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(i) for i in pivot.index])

        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Gamma")
        ax.set_title(f"Hyperparameter Sensitivity: {algo}")

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center")

        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_summary_bar(
    sum_df: pd.DataFrame,
    out_path: str,
    algorithm_col: str = "algorithm",
    value_col: str = "final_success_rate",
    ylabel: str = "Final Success Rate",
):
    stats = (
        sum_df.groupby(algorithm_col)[value_col]
        .agg(["mean", standard_error])
        .reset_index()
        .rename(columns={"standard_error": "se"})
    )

    x = np.arange(len(stats))
    plt.figure(figsize=(8, 5))
    plt.bar(x, stats["mean"], yerr=stats["se"], capsize=5)
    plt.xticks(x, stats[algorithm_col])
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by Algorithm")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main(
    episode_csv: str = "metrics/assignment2_all_algorithms_episode.csv",
    summary_csv: str = "metrics/assignment2_all_algorithms_summary.csv",
    out_dir: str = "plots",
    smooth_window: int = 10,
):
    ensure_dir(out_dir)
    ep_df, sum_df = load_metrics(episode_csv, summary_csv)

    # Optional: if random baseline was duplicated across hp loops, deduplicate for plotting
    ep_df = ep_df.drop_duplicates()
    sum_df = sum_df.drop_duplicates()

    plot_mean_se_by_algorithm(
        ep_df,
        out_path=os.path.join(out_dir, "reward_mean_se_by_algorithm.png"),
        smooth_window=smooth_window,
    )

    plot_individual_seed_overlays(
        ep_df,
        out_dir=os.path.join(out_dir, "seed_overlays"),
        smooth_window=smooth_window,
    )

    plot_hyperparameter_sensitivity(
        sum_df,
        out_path=os.path.join(out_dir, "hyperparameter_sensitivity.png"),
    )

    plot_summary_bar(
        sum_df,
        out_path=os.path.join(out_dir, "final_success_rate_bar.png"),
        value_col="final_success_rate",
        ylabel="Final Success Rate",
    )

    plot_summary_bar(
        sum_df,
        out_path=os.path.join(out_dir, "final_mean_reward_bar.png"),
        value_col="final_mean_reward",
        ylabel="Final Mean Reward",
    )

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
