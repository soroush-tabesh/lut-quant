import torch
import lut_quant

x = torch.arange(14, device='cuda', dtype=torch.bfloat16) + 0.3
lut = torch.arange(14, device='cuda', dtype=torch.bfloat16)

print("x:", x)
print("lut:", lut)
print("lut_quant.lut_quantize(x, lut):", lut_quant.lut_quantize(x, lut))
assert torch.allclose(lut_quant.lut_quantize(x, lut), lut)
