"""
merge_metrics.py  —  Combine per-agent CSVs into unified files.

Run after all three agents finish:
    python merge_metrics.py

Reads:
    metrics/random_episodes.csv     + metrics/random_summary.csv
    metrics/reinforce_episodes.csv  + metrics/reinforce_summary.csv
    metrics/dqn_episodes.csv        + metrics/dqn_summary.csv

Writes:
    metrics/assignment2_all_algorithms.csv
    metrics/assignment2_all_algorithms_summary.csv
"""

import os
import pandas as pd


def merge(output_dir: str = "metrics"):
    episode_files = [
        os.path.join(output_dir, "random_episodes.csv"),
        os.path.join(output_dir, "reinforce_episodes.csv"),
        os.path.join(output_dir, "dqn_episodes.csv"),
    ]
    summary_files = [
        os.path.join(output_dir, "random_summary.csv"),
        os.path.join(output_dir, "reinforce_summary.csv"),
        os.path.join(output_dir, "dqn_summary.csv"),
    ]

    # ── Episodes ──
    ep_frames = []
    for f in episode_files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            ep_frames.append(df)
            print(f"  Loaded {f}  ({len(df)} rows)")
        else:
            print(f"  SKIPPED {f}  (not found)")

    if ep_frames:
        ep_all = pd.concat(ep_frames, ignore_index=True)
        ep_path = os.path.join(output_dir, "assignment2_all_algorithms.csv")
        ep_all.to_csv(ep_path, index=False)
        print(f"\nMerged episodes → {ep_path}  ({len(ep_all)} total rows)")
    else:
        print("\nNo episode files found to merge.")

    # ── Summaries ──
    sum_frames = []
    for f in summary_files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            sum_frames.append(df)
            print(f"  Loaded {f}  ({len(df)} rows)")
        else:
            print(f"  SKIPPED {f}  (not found)")

    if sum_frames:
        sum_all = pd.concat(sum_frames, ignore_index=True)
        sum_path = os.path.join(output_dir, "assignment2_all_algorithms_summary.csv")
        sum_all.to_csv(sum_path, index=False)
        print(f"\nMerged summaries → {sum_path}  ({len(sum_all)} total rows)")
    else:
        print("\nNo summary files found to merge.")


if __name__ == "__main__":
    merge()
