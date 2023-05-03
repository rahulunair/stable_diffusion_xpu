import csv
import os
import random

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

        sns.barplot(
            ax=axes[1, 1], x="Configuration", y="90th Percentile Latency", data=df
        )
        axes[1, 1].set_title("90th Percentile Latency")
        axes[1, 1].set_ylabel("Latency (seconds)")
        axes[1, 1].set_xticklabels(labels=df["Configuration"], rotation=45)

        sns.barplot(
            ax=axes[2, 0], x="Configuration", y="99th Percentile Latency", data=df
        )
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


def generate_prompt(category: str) -> str:
    """
    Generate a prompt based on the given category.

    The original source of the prompt templates is from:
    https://github.com/Dalabad/stable-diffusion-prompt-templates

    Args:
        category (str): The category of the prompt.

    Returns:
        str: A generated prompt based on the category.
    """
    prompt_templates = {
        "animal": "{{Prompt}}, wildlife photography, photograph, high quality, wildlife, f 1.8, soft focus, 8k, national geographic, award - winning photograph by nick nichols",
        "archviz": "{{Prompt}}, by James McDonald and Joarc Architects, home, interior, octane render, deviantart, cinematic, key art, hyperrealism, sun light, sunrays, canon eos c 300, ƒ 1.8, 35 mm, 8k, medium - format print",
        "building": "{{Prompt}}, shot 35 mm, realism, octane render, 8k, trending on artstation, 35 mm camera, unreal engine, hyper detailed, photo - realistic maximum detail, volumetric light, realistic matte painting, hyper photorealistic, trending on artstation, ultra - detailed, realistic",
        "cartoon": "{{Prompt}}, anthro, very cute kid's film character, disney pixar zootopia character concept artwork, 3d concept, detailed fur, high detail iconic character for upcoming film, trending on artstation, character design, 3d artistic render, highly detailed, octane, blender, cartoon, shadows, lighting",
        "concept_arts": "{{Prompt}}, character sheet, concept design, contrast, style by kim jung gi, zabrocki, karlkka, jayison devadas, trending on artstation, 8k, ultra wide angle, pincushion lens effect",
    }

    prompts = {
        "animal": [
            "A lion hunting its prey",
            "A majestic elephant in the savanna",
            "A bird of paradise displaying its vibrant plumage",
            "A group of dolphins swimming in the ocean",
        ],
        "archviz": [
            "A modern open-concept living room",
            "A cozy minimalist bedroom",
            "An industrial-style kitchen",
            "A luxurious home office",
        ],
        "building": [
            "The Eiffel Tower in Paris",
            "The Burj Khalifa in Dubai",
            "The Empire State Building in New York City",
            "The Colosseum in Rome",
        ],
        "cartoon": [
            "A cute and friendly alien",
            "A brave and adventurous young knight",
            "A mischievous and clever fox",
            "A wise and gentle old wizard",
        ],
        "concept_arts": [
            "A futuristic space warrior",
            "A steampunk-inspired inventor",
            "A cybernetic detective",
            "A mythical creature with magical powers",
        ],
    }
    if category not in prompt_templates or category not in prompts:
        raise ValueError(f"Invalid category: {category}")
    template = prompt_templates[category]
    prompt = random.choice(prompts[category])
    return template.replace("{{Prompt}}", prompt)
