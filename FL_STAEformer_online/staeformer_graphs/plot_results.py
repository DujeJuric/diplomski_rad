import os
import glob
import json
import numpy as np

def generate_comparison_plot():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    graphs_dir = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(graphs_dir, "staeformer_client_*_metrics.json"))
    if not files:
        print(f"No metrics files found inside '{graphs_dir}'.")
        return

    client_data = {}
    for f in files:
        try:
            filename = os.path.basename(f)
            client_id = int(filename.split("_")[1])
        except ValueError:
            client_id = f
            
        with open(f, "r") as file_handle:
            client_data[client_id] = json.load(file_handle)

    sorted_client_ids = sorted(list(client_data.keys()))
    
    first_client = sorted_client_ids[0]
    steps = client_data[first_client]["step"]
    
    metrics_keys = ["loss", "mae", "rmse", "mape"]
    metric_labels = {
        "loss": "Loss",
        "mae": "MAE",
        "rmse": "RMSE",
        "mape": "MAPE"
    }
    
    averages = {k: np.zeros(len(steps)) for k in metrics_keys}
    for k in metrics_keys:
        for cid in sorted_client_ids:
            averages[k] += np.array(client_data[cid][k])
        averages[k] /= len(sorted_client_ids)

    fig, axs = plt.subplots(2, 2, figsize=(14, 11))

    for idx, key in enumerate(metrics_keys):
        row = idx // 2
        col = idx % 2
        ax = axs[row, col]
        
        ax.plot(
            steps, 
            averages[key], 
            marker='o', 
            linestyle='-', 
            color='blue', 
            linewidth=2, 
            label="Federated Average"
        )
        
        ax.set_title(metric_labels[key], fontsize=12, fontweight="bold")
        ax.set_xlabel("Online Step", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_filename = os.path.join(graphs_dir, "federated_metrics_comparison.png")
    plt.savefig(output_filename, dpi=200)
    plt.close()

if __name__ == "__main__":
    generate_comparison_plot()
