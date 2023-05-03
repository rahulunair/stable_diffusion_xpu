import os
import warnings

warnings.filterwarnings("ignore")

import logging

import intel_extension_for_pytorch as ipex
import torch
from diffusers import DPMSolverMultistepScheduler

from sd_xpu import ModelConfig, run_experiment
from utils import plot_latency_results
from utils import save_results_to_csv
from utils import mkdirs

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
    

def main():
    results_path = mkdirs("results")
    setup_logging(results_path)
    models = ["stabilityai/stable-diffusion-2-1", "runwayml/stable-diffusion-v1-5"]
    prompts = [
        "A warrior from D&D holding his weapon backwards while looking at the battlefield, oil painting, portrait, armored, high res"
    ]
    model_config = ModelConfig(
        model_id=models[1],
        prompt=prompts[0],
    )
    devices = ["cpu", "xpu"]
    dmodes = [
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
