import torch 
import torch.nn as nn 
from otdd.pytorch.utils import generate_and_plot_data

num_channels = 3
num_projection = 10000

U1 = nn.Conv2d(num_channels, num_projection, kernel_size=5, stride=2, padding=0, bias=False)
U2 = nn.Conv2d(num_projection, num_projection, kernel_size=5, stride=2, padding=0, bias=False, groups=num_projection)
U3 = nn.Conv2d(num_projection, num_projection, kernel_size=3, stride=2, padding=0, bias=False, groups=num_projection)
U4 = nn.Conv2d(num_projection, num_projection, kernel_size=2, stride=1, padding=0, bias=False, groups=num_projection)
U_list = [U1, U2, U3, U4]

# U1 = nn.Conv2d(num_channels, num_projection, kernel_size=5, stride=1, padding=0, bias=False)
# U2 = nn.Conv2d(num_projection, num_projection, kernel_size=5, stride=1, padding=0, bias=False, groups=num_projection)
# U3 = nn.Conv2d(num_projection, num_projection, kernel_size=5, stride=2, padding=0, bias=False, groups=num_projection)
# U4 = nn.Conv2d(num_projection, num_projection, kernel_size=5, stride=2, padding=0, bias=False, groups=num_projection)
# U5 = nn.Conv2d(num_projection, num_projection, kernel_size=3, stride=1, padding=0, bias=False, groups=num_projection)
# U_list = [U1, U2, U3, U4, U5]

for U in U_list:
    U.weight.data = torch.randn(U.weight.data.shape)
    U.weight.data = U.weight.data / torch.sqrt(torch.sum(U.weight.data ** 2,dim=[1,2,3],keepdim=True))
    U.requires_grad = False

x = torch.randn(1, 3, 32, 32)
generate_and_plot_data(x[0], "cac1.png")
print(x.min(), x.max())

for conv in U_list:
    x = conv(x)
generate_and_plot_data(x[0], "cac2.png")

print(x.shape)
print(x.min(), x.max())
