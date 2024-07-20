import torch
import random

# k = torch.arange(start=1, end=6, step=1)
# print(k)
# num_examples = 3
# num_projection = 4
# x = torch.randint(low=1, high=5, size=(num_examples, num_projection))
# print(x)
# # x_repeat = torch.expand()
# pow_x = torch.pow(input=x.permute(1, 0).unsqueeze(-1), exponent=k).permute(0, 1, 2)
# print(pow_x, pow_x.shape)


x = torch.randn(3, 4)
y = torch.randn(3, 2)
print(x)
print(y)
print(y.unsqueeze(1).expand(3, 4, 2))
z = torch.stack((x, y.unsqueeze(1).expand(3, 4, 2)), dim=2)
print(z, z.shape)