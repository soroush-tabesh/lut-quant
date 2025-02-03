import torch
import lut_quant

x = torch.arange(16, device='cuda', dtype=torch.bfloat16) + 0.3
lut = torch.arange(16, device='cuda', dtype=torch.bfloat16)

print("x:", x)
print("lut:", lut)
print("lut_quant.lut_quantize(x, lut):", lut_quant.lut_quantize(x, lut))
