import os 
from src.support.reusable import LearningCurvesComparer , epoch_time_comparison, framework_vs_final_acc

def main():
    out_dir = "viz/learning_curves"
    os.makedirs(out_dir, exist_ok=True)

    iter_csv_paths = {
        "jax": "metrics/Assignment1_JAX_dataset_metrics.csv",
        "torch": "metrics/Assignment1_torch_dataset_metrics.csv",
    }

    plotter = LearningCurvesComparer(iter_csv_paths)

    for fw in ["jax", "torch"]:
        for metric in ["Loss", "Accuracy"]:
            # iteration vs seed (all curves)
            plotter.plot_iteration_vs_col(framework=fw, metric=metric, ref_col="Seed", out_dir=out_dir)

            # iteration vs batch_size (mean±SD and mean±SE)
            plotter.plot_iteration_vs_col(framework=fw, metric=metric, ref_col="Batch_Size", out_dir=out_dir)
            epoch_time_comparison(summary_csv="results/summary.csv",output_dir="viz")
            framework_vs_final_acc(summary_csv="results/summary.csv",output_dir="viz")
    print("Saved plots to:", out_dir)

if __name__ == "__main__":
    main()
