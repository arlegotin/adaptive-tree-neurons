import torch
from treeneuron.models.units import SelectiveAffineLayer

torch.manual_seed(42)
layer = SelectiveAffineLayer(d=16, h=64)
x = torch.randn(32, 16)
before = layer(x).detach()

# Contrast rows contain 16 weights and a bias. New gates start at zero.
manifest = layer.grow(torch.arange(16), 0.05 * torch.randn(16, 17))
after = layer(x)

torch.testing.assert_close(after, before, rtol=0, atol=0)
print(manifest.added_parameter_count)  # 544
print(after.relu().shape)             # torch.Size([32, 64])
