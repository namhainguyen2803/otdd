import torch
# a = 2
# b = 3

# z_concat = list()
# for i in range(2):
#     x = torch.randint(10, size=(a,))
#     y = torch.randint(4, size=(a, b))
#     print(f"x: {x}")
#     print(f"y: {y}")
#     z = torch.stack((y, x.unsqueeze(1).expand(-1, b)), dim=2)
#     print(f"z: {z}")
#     z_concat.append(z)

# z_concat = torch.cat(z_concat, dim=1)

# print(z_concat)

x = torch.randn(3, 4, 2)
y = torch.randn(3, 2)
z = torch.matmul(x, y.unsqueeze(-1)).squeeze(-1)
print(x)
print(y)

print(z, z.shape)