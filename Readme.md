### Stable diffusion benchmark for Intel discrete GPUs

To setup the environment with PyTorch for Intel GPUs (Arc, Flex, Max) and other deps, run:

```bash
conda env create -f environment.yml
```

## Test the GPU

```bash
conda activate test2
python main.py --device xpu --random
```
