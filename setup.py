from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import pathlib
import torch

setup_dir = os.path.dirname(os.path.realpath(__file__))

def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = [
        '-D__CUDA_NO_HALF_OPERATORS__',
        '-D__CUDA_NO_HALF_CONVERSIONS__',
        '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '-D__CUDA_NO_HALF2_OPERATORS__',
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            torch.utils.cpp_extension.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass

def get_cuda_arch_flags():
    return [
        '-gencode', 'arch=compute_80,code=sm_80',  # Ampere
        '-gencode', 'arch=compute_86,code=sm_86',  # Ampere
        '-gencode', 'arch=compute_89,code=sm_89',  # Ada
        '-gencode', 'arch=compute_90,code=sm_90',  # Hopper
        '--expt-relaxed-constexpr'
    ]

if __name__ == '__main__':
    assert torch.cuda.is_available(), "CUDA is not available!"
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    assert capability[0] >= 8, f"CUDA capability must be >= 8.0, yours is {capability}"

    remove_unwanted_pytorch_nvcc_flags()
    setup(
        name='lut_quant',
        ext_modules=[
            CUDAExtension(
                name='lut_quant._CUDA',
                sources=[
                    'lut_quant/kernels/bindings.cpp',
                    'lut_quant/kernels/codebook_quant_bf16_fast.cu',
                    'lut_quant/kernels/codebook_quant_fp32_fast.cu',
                ],
                include_dirs=[
                    os.path.join(setup_dir, 'lut_quant/kernels/include'),
                ],
                extra_compile_args={
                    'cxx': [],
                    'nvcc': get_cuda_arch_flags(),
                }
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        },
        version='0.1.0',
    )