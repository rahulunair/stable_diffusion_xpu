import csv
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def mkdirs(name: str):
    path = os.path.join(os.getcwd(), name)
    os.makedirs(path, exist_ok=True)
    return path

def plot_latency_results(configurations, latencies):
    data = {
        "Configuration": [],
        "Average Latency": [],
        "Mean Latency": [],
        "Median Latency": [],
        "Standard Deviation": [],
        "90th Percentile Latency": [],
        "99th Percentile Latency": [],
    }
    for config, latency in zip(configurations, latencies):
        data["Configuration"].append(config)
        data["Average Latency"].append(latency["average_latency"])
        data["Mean Latency"].append(latency["mean_latency"])
        data["Median Latency"].append(latency["median_latency"])
        data["Standard Deviation"].append(latency["stdev_latency"])
        data["90th Percentile Latency"].append(latency["90th_percentile_latency"])
        data["99th Percentile Latency"].append(latency["99th_percentile_latency"])
    df = pd.DataFrame(data)
    with plt.xkcd():
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle("Stable diffusion latency results")
        sns.barplot(ax=axes[0, 0], x="Configuration", y="Mean Latency", data=df)
        axes[0, 0].set_title("Mean Latency")
        axes[0, 0].set_ylabel("Latency (seconds)")
        axes[0, 0].set_xticklabels(labels=df["Configuration"], rotation=45)
        
        sns.barplot(ax=axes[0, 1], x="Configuration", y="Median Latency", data=df)
        axes[0, 1].set_title("Median Latency")
        axes[0, 1].set_ylabel("Latency (seconds)")
        axes[0, 1].set_xticklabels(labels=df["Configuration"], rotation=45)

        sns.barplot(ax=axes[1, 0], x="Configuration", y="Standard Deviation", data=df)
        axes[1, 0].set_title("Standard Deviation")
        axes[1, 0].set_ylabel("Latency (seconds)")
        axes[1, 0].set_xticklabels(labels=df["Configuration"], rotation=45)

        sns.barplot(ax=axes[1, 1], x="Configuration", y="90th Percentile Latency", data=df)
        axes[1, 1].set_title("90th Percentile Latency")
        axes[1, 1].set_ylabel("Latency (seconds)")
        axes[1, 1].set_xticklabels(labels=df["Configuration"], rotation=45)

        sns.barplot(ax=axes[2, 0], x="Configuration", y="99th Percentile Latency", data=df)
        axes[2, 0].set_title("99th Percentile Latency")
        axes[2, 0].set_ylabel("Latency (seconds)")
        axes[2, 0].set_xticklabels(labels=df["Configuration"], rotation=45)
        
        sns.boxplot(ax=axes[2, 1], x="Configuration", y="Average Latency", data=df)
        axes[2, 1].set_title("Average Latency Distribution")
        axes[2, 1].set_ylabel("Latency (seconds)")
        axes[2, 1].set_xticklabels(labels=df["Configuration"], rotation=45)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig("sd_latency_plot.png", bbox_inches="tight")


def save_results_to_csv(configurations, latencies, filename="sd_latency.csv"):
    results_path = mkdirs("results")
    data = {
        "Configuration": configurations,
        "Average Latency": [],
        "Mean Latency": [],
        "Median Latency": [],
        "Standard Deviation": [],
        "90th Percentile Latency": [],
        "99th Percentile Latency": [],
    }
    for latency in latencies:
        data["Average Latency"].append(latency["average_latency"])
        data["Mean Latency"].append(latency["mean_latency"])
        data["Median Latency"].append(latency["median_latency"])
        data["Standard Deviation"].append(latency["stdev_latency"])
        data["90th Percentile Latency"].append(latency["90th_percentile_latency"])
        data["99th Percentile Latency"].append(latency["99th_percentile_latency"])
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(results_path, filename), index=False)
