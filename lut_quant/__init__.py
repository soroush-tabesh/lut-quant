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
        input: Input tensor to be quantized
        lut: Lookup table tensor containing possible quantization values
        
    Returns:
        Quantized tensor with values from the lookup table
        
    Raises:
        ValueError: If input dtype is not bfloat16 or float32
        ValueError: If input and lut have different dtypes or devices
        ValueError: If lut is not 1D or has more than 256 elements
    '''
    # Input validation
    if input.dtype not in [torch.bfloat16, torch.float32]:
        raise ValueError(f"Input dtype must be bfloat16 or float32, got {input.dtype}")
        
    if input.dtype != lut.dtype:
        raise ValueError(f"Input and lookup table must have same dtype, got {input.dtype} and {lut.dtype}")
        
    if input.device != lut.device:
        raise ValueError("Input and lookup table must be on same device")
        
    if lut.ndim != 1:
        raise ValueError(f"Lookup table must be 1D, got {lut.ndim}D")
        
    if len(lut) > 256:
        raise ValueError(f"Lookup table must have ≤256 elements, got {len(lut)}")

    # Pad lookup table to nearest supported size if needed
    SUPPORTED_SIZES = [4, 8, 16, 64, 256]
    if len(lut) not in SUPPORTED_SIZES:
        target_size = next(size for size in SUPPORTED_SIZES if size >= len(lut))
        padding = target_size - len(lut)
        lut = torch.cat([lut, torch.full((padding,), lut[-1], dtype=lut.dtype, device=lut.device)])

    # Dispatch to appropriate CUDA kernel
    if input.dtype == torch.bfloat16:
        return lut_quant._CUDA.codebook_quantize(input, lut)
    elif input.dtype == torch.float32:
        return lut_quant._CUDA.codebook_quantize_f(input, lut)
    else:
        raise ValueError(f"Unsupported data type: {input.dtype}, only bfloat16 and float32 are supported")


__all__ =  ["lut_quantize"]
