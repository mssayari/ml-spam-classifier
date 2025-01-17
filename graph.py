import pandas as pd
import matplotlib.pyplot as plt
import json

# File names
files = [
    "results/ai_dataset_results.json",
    "results/combined_dataset_results.json",
    "results/completeSpamAssassin_results.json"
]
file_labels = ["AI Dataset", "Combined Dataset", "Complete SpamAssassin"]

# Load data from files into a dictionary of DataFrames
dataframes = {}
for file, label in zip(files, file_labels):
    with open(file, "r") as f:
        data = json.load(f)
        dataframes[label] = pd.DataFrame(data).T

# Combine the data into a single DataFrame for comparison
combined_df = pd.concat(dataframes, keys=file_labels, names=["Dataset", "Model"])

# Save the combined data as a CSV file (optional)
combined_df.to_csv("combined_results.csv")

# Function to plot comparison charts
def plot_comparison(metric, combined_df):
    metric_df = combined_df[metric].unstack(level=0)  # Separate datasets into columns
    metric_df.plot(kind="bar", figsize=(10, 6), colormap="viridis", edgecolor="black")
    # plt.title(f"{metric} Comparison Across Datasets", fontsize=16)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.legend(title="Dataset", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{metric}_comparison.png")
    plt.close()

# Plot Precision, Recall, and F1-Score comparisons
metrics = ["Precision", "Recall", "F1-Score"]
for metric in metrics:
    plot_comparison(metric, combined_df)

# Generate line plots to observe trends
def plot_trend(metric, combined_df):
    metric_df = combined_df[metric].unstack(level=0)
    metric_df.plot(figsize=(10, 6), marker="o")
    plt.title(f"{metric} Trend Across Models and Datasets", fontsize=16)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.legend(title="Dataset", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{metric}_trend.png")
    plt.close()

for metric in metrics:
    plot_trend(metric, combined_df)

print("Comparison charts and trend graphs have been generated and saved as images.")