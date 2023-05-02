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
        "Configuration": configurations,
        "Latency": latencies,
    }
    df = pd.DataFrame(data)
    with plt.xkcd():
        plt.figure(figsize=(12, 6))
        sns.barplot(x="Configuration", y="Latency", data=df)
        plt.title("Stable diffusion latency results.")
        plt.xlabel("Configuration")
        plt.ylabel("Latency (seconds)")
        plt.xticks(rotation=45)
        plt.savefig("stable_diffussion_latency_results.png", bbox_inches="tight")
        plt.show()


def save_results_to_csv(configurations, latencies, filename="sd_latency.csv"):
    results_path = mkdirs("results")
    with open(os.path.join(results_path,filename), mode="w", newline="") as csvfile:
        fieldnames = ["config", "latency"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for config, latency in zip(configurations, latencies):
            writer.writerow({"config": config, "latency": latency})
