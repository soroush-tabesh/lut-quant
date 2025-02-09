# LUT-Quant

A CUDA-accelerated lookup table quantization library for PyTorch.

## Requirements

- CUDA-capable GPU with compute capability >= 8.0 (Ampere/Ada/Hopper)
- PyTorch
- Ninja build system

## Installation

Make sure that CUDA is installed and the environment variables `CUDA_HOME` and `LD_LIBRARY_PATH` are set.
The library has been tested on Ubuntu 22.04 with CUDA 12.4.

1. Install the requirements:

```bash
pip install -r requirements.txt
```

2. Build the library:

```bash
pip install .
```

## Usage

```python
import lut_quant

x = torch.randn(10, 10, device='cuda', dtype=torch.bfloat16)
lut = torch.tensor([-1.5, -0.5, 0.0, 0.5, 1.5], device='cuda', dtype=torch.bfloat16)

print(lut_quant.lut_quantize(x, lut))
```
