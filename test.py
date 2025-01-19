import torch
import math
from otdd.pytorch.utils import generate_moments


print(torch.arange(1) + 1)

cac = generate_moments(num_moments=5, min_moment=1, max_moment=8)

print(cac)