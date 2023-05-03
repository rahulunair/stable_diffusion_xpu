import time
from dataclasses import dataclass

import intel_extension_for_pytorch as ipex
import torch
from diffusers import StableDiffusionPipeline

from utils import mkdirs


@dataclass
class ModelConfig:
    model_id: str
    prompt: str
    nb_pass: int = 10
    num_inference_steps: int = 20
    prefix: str = ""


def load_pipeline(model_config: ModelConfig, device: str, scheduler=None):
    if scheduler is None:
        pipe = StableDiffusionPipeline.from_pretrained(model_config.model_id)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_config.model_id, scheduler=scheduler
        )
    return pipe.to(device)


def apply_ipex_optimize(pipe, dtype, input_example=None):
    pipe.unet = ipex.optimize(
        pipe.unet.eval(), dtype=dtype, inplace=True, sample_input=input_example
    )
    pipe.vae = ipex.optimize(pipe.vae.eval(), dtype=dtype, inplace=True)
    pipe.text_encoder = ipex.optimize(
        pipe.text_encoder.eval(), dtype=dtype, inplace=True
    )
    if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
        pipe.safety_checker = ipex.optimize(
            pipe.safety_checker.eval(), dtype=dtype, inplace=True
        )
    return pipe


def apply_memory_format_optimization(pipe):
    pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
    pipe.text_encoder = pipe.text_encoder.to(memory_format=torch.channels_last)
    if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
        pipe.safety_checker = pipe.safety_checker.to(memory_format=torch.channels_last)
    return pipe


def elapsed_time(
    pipeline: StableDiffusionPipeline, config: ModelConfig, device="cpu"
) -> float:
    images = pipeline(
        config.prompt, num_inference_steps=ModelConfig.num_inference_steps
    ).images
    start = time.time()
    for _ in range(config.nb_pass):
        images = pipeline(
            config.prompt, num_inference_steps=config.num_inference_steps
        ).images
        if device.startswith("xpu"):
            torch.xpu.synchronize()
    end = time.time()
    images_path = mkdirs("results/images")
    images[0].save(f"{images_path}/{config.prompt.split()[0]}_{config.prefix}.png")
    return (end - start) / config.nb_pass


def run_experiment(
    model_config: ModelConfig,
    device: str,
    dtype=None,
    scheduler=None,
    ipex_optimize=False,
):
    pipe = load_pipeline(model_config, device, scheduler)
    if ipex_optimize:
        input_example = (
            torch.randn(2, 4, 64, 64),
            torch.rand(1) * 999,
            torch.randn(2, 77, 768),
        )
        pipe = apply_memory_format_optimization(pipe)
        pipe = apply_ipex_optimize(pipe, dtype, input_example)
    if dtype and dtype != torch.float32:
        if device.startswith("xpu"):
            with torch.xpu.amp.autocast(enabled=True, dtype=dtype):
                latency, loop_time = elapsed_time(pipe, model_config, device=device)
        else:
            with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                latency, loop_time = elapsed_time(pipe, model_config, device=device)
    else:
        latency, loop_time = elapsed_time(pipe, model_config, device=device)
    mean_latency = statistics.mean(loop_time)
    median_latency = statistics.median(loop_time)
    stdev_latency = statistics.stdev(loop_time)
    percentile_90 = np.percentile(loop_time, 90)
    percentile_99 = np.percentile(loop_time, 99)

    return {
        "average_latency": latency,
        "mean_latency": mean_latency,
        "median_latency": median_latency,
        "stdev_latency": stdev_latency,
        "90th_percentile_latency": percentile_90,
        "99th_percentile_latency": percentile_99,
    }
