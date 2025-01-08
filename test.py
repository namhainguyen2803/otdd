import torch
import math


cac = torch.sort(torch.randperm(10)[:5])[0]
print(cac)
cac[cac < 1] = 1
print(cac)