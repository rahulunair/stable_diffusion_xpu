import os
import warnings

warnings.filterwarnings("ignore")

import logging

import intel_extension_for_pytorch as ipex
import torch
from diffusers import DPMSolverMultistepScheduler

from sd_xpu import ModelConfig, run_experiment
from utils import plot_latency_results, save_results_to_csv

os.environ["OMP_NUM_THREADS"] = "56"
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["OMP_DYNAMIC"] = "false"
os.environ["KMP_AFFINITY"] = "compact,granularity=fine,1,0"

torch.manual_seed(12345)
ipex.xpu.seed_all()


def setup_logging():
    logging.basicConfig(filename="performance.log", level=logging.INFO)


def main():
    setup_logging()
    models = ["stabilityai/stable-diffusion-2-1", "runwayml/stable-diffusion-v1-5"]
    prompts = [
        "A warrior from D&D holding his weapon backwards while looking at the battlefield, oil painting, portrait, armored, high res"
    ]
    model_config = ModelConfig(
        model_id=models[1],
        prompt=prompts[0],
    )
    devices = ["xpu", "cpu"]
    modes = [
        {"dtype": torch.bfloat16, "ipex_optimize": True, "scheduler": True},
        {"dtype": torch.bfloat16, "ipex_optimize": True, "scheduler": True},
        {"dtype": torch.float32, "ipex_optimize": False, "scheduler": False},
        {"dtype": torch.float32, "ipex_optimize": True, "scheduler": False},
        {"dtype": torch.float32, "ipex_optimize": True, "scheduler": True},
    ]
    configs = []
    latencies = []
    for device in devices:
        if device == "xpu":
            modes = [
                {"dtype": torch.float16, "ipex_optimize": True, "scheduler": False},
                {"dtype": torch.float16, "ipex_optimize": True, "scheduler": True},
            ] + modes
        else:
            modes = modes
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
            latency = run_experiment(
                model_config, device, dtype, scheduler, ipex_optimize
            )
            print(f"Latency for {device} with mode {model_config.prefix}: {latency}")
            logging.info(
                f"Latency for {device} with mode {model_config.prefix}: {latency}"
            )
            configs.append(model_config.prefix)
            latencies.append(latency)
    save_results_to_csv(configs, latencies)
    plot_latency_results(configs, latencies)


if __name__ == "__main__":
    main()
