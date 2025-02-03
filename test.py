import torch
import lut_quant

x = torch.arange(10, device='cuda', dtype=torch.bfloat16)
lut = torch.tensor([0.0, 5.0, 10.0], device='cuda', dtype=torch.bfloat16)

print("x:", x)
print("lut:", lut)
print("lut_quant.lut_quantize(x, lut):", lut_quant.lut_quantize(x, lut))
