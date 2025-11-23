from matplotlib import pyplot as plt

def plot_all(datasets, results_acc, results_f1, k):
    """
    Automatically generate 6 plots:
      - Accuracy for each ML model
      - F1 for each ML model
    """

    models = {
        "Logistic Regression": ("lr", "LR"),
        "Random Forest": ("rf", "RF"),
        "LightGBM": ("lg", "LGBM")
    }

    metrics = {
        "Accuracy": results_acc,
        "F1": results_f1
    }

    def plot_single(model_name, results, metric_name):
        plt.figure(figsize=(10, 6))
        for dataset in datasets:
            plt.plot(
                k,
                results[model_name][dataset],
                marker='o',
                linewidth=2,
                markersize=7,
                label=dataset.capitalize()
            )

        plt.xlabel("Group Size", fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.title(f"{metric_name} vs Group Size — {model_name}", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.xticks(k)
        plt.legend(fontsize=10)
        plt.tight_layout()

        filename = f"{metric_name.lower()}_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Saved: {filename}")

    # Main loop for all models & metrics
    for model_name in models:
        for metric_name in metrics:
            plot_single(model_name, metrics[metric_name], metric_name)
