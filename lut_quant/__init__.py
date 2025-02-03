import torch
import lut_quant._CUDA


def lut_quantize(
        input: torch.Tensor,
        lut: torch.Tensor,
        ) -> torch.Tensor:
    '''
    Quantize a tensor using a lookup table. `lut` is a tensor of possible values for the input tensor. 
    The input tensor is quantized to the nearest value in the lookup table. Tensors have to be of the same dtype and device.
    Args:
        input: torch.Tensor, shape (M, K)
        lut: torch.Tensor, shape (K, N) 
    Returns:
        torch.Tensor, shape (M, N)
    '''
    assert input.dtype == lut.dtype, "Input and lookup table must have the same dtype"
    assert input.device == lut.device, "Input and lookup table must be on the same device"
    if input.dtype == torch.bfloat16:
        return lut_quant._CUDA.codebook_quantize(input, lut)
    elif input.dtype == torch.float32:
        return lut_quant._CUDA.codebook_quantize_f(input, lut)
    else:
        raise ValueError(f"Unsupported data type: {input.dtype}, only bfloat16 and float32 are supported")


__all__ =  ["lut_quantize"]
