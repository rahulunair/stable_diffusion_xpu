import argparse
import os
import random
import warnings


warnings.filterwarnings("ignore")

import logging

import intel_extension_for_pytorch as ipex
import torch
from diffusers import DPMSolverMultistepScheduler

from sd_xpu import ModelConfig, run_experiment
from utils import (generate_prompt, mkdirs, plot_latency_results,
                   save_results_to_csv)

os.environ["OMP_NUM_THREADS"] = "56"
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["OMP_DYNAMIC"] = "false"
os.environ["KMP_AFFINITY"] = "compact,granularity=fine,1,0"

torch.manual_seed(12345)
ipex.xpu.seed_all()


def setup_logging(results_path=None, log_filename="latency.log"):
    if results_path is not None:
        log_path = os.path.join(results_path, log_filename)
    else:
        log_path = log_filename
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run stable diffusion with different configurations."
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "xpu", "both"],
        default="both",
        help="Specify the device to run on (cpu, xpu, or both).",
    )
    parser.add_argument(
        "--random", action="store_true", help="Generate a random prompt"
    )
    return parser.parse_args()


def main():
    results_path = mkdirs("results")
    setup_logging(results_path)

    model = "runwayml/stable-diffusion-v1-5"
    dmodes = [
        {"dtype": torch.bfloat16, "ipex_optimize": True, "scheduler": False},
        {"dtype": torch.bfloat16, "ipex_optimize": True, "scheduler": True},
        {"dtype": torch.float32, "ipex_optimize": False, "scheduler": False},
        {"dtype": torch.float32, "ipex_optimize": True, "scheduler": False},
        {"dtype": torch.float32, "ipex_optimize": True, "scheduler": True},
    ]
    configs = []
    latencies = []
    args = parse_args()
    if args.random:
        category = random.choice(
            ["animal", "archviz", "cartoon", "building", "concept_art"]
        )
        prompts = [f"{generate_prompt(category)}"]
    else:
        prompts = [
            "Portrait photo of a Ottoman market at night with no people during the golden age, photograph, depth of field, moody light, golden hour, inspired by the style of asiatic paintings, extremely detailed, taken with a Nikon D850, award-winning photography"
        ]
    model_config = ModelConfig(
        model_id=model,
        prompt=prompts[0],
    )
    device_arg = args.device
    if device_arg == "both":
        devices = ["cpu", "xpu"]
    else:
        devices = [device_arg]
    for device in devices:
        if device == "xpu":
            modes = [
                {"dtype": torch.float16, "ipex_optimize": True, "scheduler": False},
                {"dtype": torch.float16, "ipex_optimize": True, "scheduler": True},
            ] + dmodes
        else:
            modes = dmodes
        for mode in modes:
            dtype = mode["dtype"]
            ipex_optimize = mode["ipex_optimize"]
            use_scheduler = mode["scheduler"]
            prefix = []
            if dtype:
                prefix.append(dtype.__str__().split(".")[-1])
            if ipex_optimize:
                prefix.append("ipex")
            if use_scheduler:
                prefix.append("sched")
            prefix.append(device)
            model_config.prefix = "_".join(prefix)
            print(f"Running experiment on {device} with mode {model_config.prefix}")
            if use_scheduler:
                scheduler = DPMSolverMultistepScheduler.from_pretrained(
                    model_config.model_id, subfolder="scheduler"
                )
            else:
                scheduler = None
            latency_stats = run_experiment(
                model_config, device, dtype, scheduler, ipex_optimize
            )
            device_and_mode = f"{device} with mode {model_config.prefix}"
            logging.info(f"Latency statistics for {device_and_mode}:")
            logging.info(f"  Average Latency: {latency_stats['average_latency']:.6f}")
            logging.info(f"  Mean Latency: {latency_stats['mean_latency']:.6f}")
            logging.info(f"  Median Latency: {latency_stats['median_latency']:.6f}")
            logging.info(f"  Standard Deviation: {latency_stats['stdev_latency']:.6f}")
            logging.info(
                f"  90th Percentile Latency: {latency_stats['90th_percentile_latency']:.6f}"
            )
            logging.info(
                f"  99th Percentile Latency: {latency_stats['99th_percentile_latency']:.6f}"
            )

            configs.append(model_config.prefix)
            latencies.append(latency_stats)
    save_results_to_csv(configs, latencies)
    plot_latency_results(configs, latencies)


if __name__ == "__main__":
    main()
