import torch
from otdd.pytorch.utils import generate_unit_convolution_projections, generate_uniform_unit_sphere_projections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader



# num_moments = 3
# num_examples = 4
# num_projection = 2
# moments = torch.randint(1, num_moments + 1, (num_projection, num_moments))

# x = torch.randint(1, 5, (num_examples, num_projection))
# x_pow = torch.pow(x.unsqueeze(1), moments.permute(1, 0)).permute(2, 0, 1)


# print(moments)

# print(x.permute(1, 0))

# print(x_pow, x_pow.shape)

# x = torch.randn(1, 1, 28, 28).to('cpu')

# U_list = generate_unit_convolution_projections(image_size=28, num_channels=1, num_projection=100, device='cpu')

# for U in U_list:
#     x = U(x)

# print(x.shape)

# proj_matrix_dataset = list()
# for i in range(2):
#     X_projection = torch.randint(1, 4, size=(NUM_EXAMPLES, NUM_PROJECTION))
#     k = torch.randint(1, 4, size=(NUM_PROJECTION, NUM_MOMENTS))
#     print(X_projection.t())
#     print(k)
#     moment_X_projection = torch.pow(input=X_projection.unsqueeze(1), exponent=k.permute(1, 0))
#     print(moment_X_projection)
#     moment_X_projection = moment_X_projection.permute(2, 0, 1) 
#     print(moment_X_projection)
#     avg_moment_X_projection = torch.sum(moment_X_projection, dim=1)
#     print(avg_moment_X_projection)

#     X_projection = torch.permute(X_projection, dims=(1, 0)) # shape == (num_projection, num_examples)
#     h = torch.cat((X_projection.unsqueeze(-1),
#                     avg_moment_X_projection.unsqueeze(1).expand(NUM_PROJECTION, NUM_EXAMPLES, NUM_MOMENTS)), 
#                     dim=2) 
#     # shape == (num_projection, num_examples, num_moments+1)
#     print(h.shape)
#     print(h)
#     # print(h.permute(1,0,2))
#     proj_matrix_dataset.append(h)
#     print("---------")

# proj_matrix_dataset = torch.cat(proj_matrix_dataset, dim=1) 
# print(proj_matrix_dataset)

# projection_matrix_2 = torch.randn(NUM_PROJECTION, NUM_MOMENTS+1)
# proj_matrix_dataset = proj_matrix_dataset.type(torch.FloatTensor)
# proj_proj_matrix_dataset = torch.matmul(proj_matrix_dataset, projection_matrix_2.unsqueeze(-1)).squeeze(-1) # shape == (num_projection, total_examples)

# print("==========")
# print(proj_proj_matrix_dataset)
# print("=========")

# for i in range(NUM_PROJECTION):
#     print(torch.matmul(proj_matrix_dataset[i, :, :], projection_matrix_2[i, :]))




NUM_EXAMPLES = 3
NUM_PROJECTION = 4
NUM_PROJECTION_2 = 4
NUM_MOMENTS = 2
DIM = 5
# proj_matrix_dataset = torch.randn(NUM_PROJECTION, NUM_EXAMPLES, NUM_MOMENTS + 1)

# projection_matrix = torch.randn((NUM_PROJECTION, NUM_PROJECTION_2, NUM_MOMENTS + 1))
# projection_matrix = projection_matrix / torch.sqrt(torch.sum(projection_matrix ** 2, dim=2, keepdim=True))

# print(proj_matrix_dataset.shape, projection_matrix.shape)
# x = torch.matmul(proj_matrix_dataset, projection_matrix.unsqueeze(-1))

# result = list()
# for i in range(NUM_PROJECTION):
#     a = proj_matrix_dataset[i, :, :]
#     b = projection_matrix[i, :, :]
#     c = torch.matmul(a, b.permute(1, 0))
#     c = torch.sum(c, dim=1) / 
#     print(c.shape)
#     result.append(c.unsqueeze(0))
# result = torch.concat(result, dim=0)
# print(result.shape)


# x = torch.randn((NUM_PROJECTION, DIM))
# print(x)
# y = torch.zeros(NUM_PROJECTION, DIM)
# ind = torch.randperm(NUM_PROJECTION)[:2]
# y[ind, 0] = 1
# print(y)
# print(x+y)


# tensor = torch.randn((NUM_PROJECTION, DIM))
# tensor2 = tensor.clone()
# print("Original Tensor:\n", tensor)

# # Find the maximum value and its index along each row
# max_vals, max_indices = torch.max(tensor2, dim=1)
# print(max_vals)
# tmp = tensor2[:, 0].clone()
# tensor2[:, 0] = max_vals
# tensor2[range(NUM_PROJECTION), max_indices] = tmp

# print(tensor2)

# # Swap the first entry with the maximum if it's not already the largest
# for i in range(NUM_PROJECTION):
#     if max_indices[i] != 0:  # If the max isn't already in the first position
#         # Swap the first entry with the max entry
#         tensor[i, 0], tensor[i, max_indices[i]] = tensor[i, max_indices[i]], tensor[i, 0]

# print("\nModified Tensor:\n", tensor)




# # Define transformations
# transform = transforms.Compose([
#     transforms.ToTensor(),         # Convert images to tensors
#     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
# ])

# # Load the dataset
# dataset = CIFAR100(root='data2/CIFAR100', download=True, transform=transform)

# # Create a DataLoader
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # Example: Iterate through the DataLoader
# for images, labels in dataloader:
#     # Your training code here
#     print(images.shape)
#     break

# print(images[0], labels[0])

parent_dir = "saved/split_cifar100"

nm_saved_dir = f"{parent_dir}/new_method_dist.pt"
nm_tensor = torch.load(nm_saved_dir)
print(nm_tensor)

otdd_saved_dir = f"{parent_dir}/otdd_dist.pt"
otdd_tensor = torch.load(otdd_saved_dir)
print(otdd_tensor)

list_nm = list()
list_otdd = list()
for i in range(len(nm_tensor)):
    for j in range(i+1, len(nm_tensor)):
        list_nm.append(nm_tensor[i,j].item())
        list_otdd.append(otdd_tensor[i,j].item())


rho, p_value = stats.pearsonr(list_otdd, list_nm)
print(f"Overall Pearson correlation coefficient: {rho}")
print(f"Overall P-value: {p_value}")

with open(f'{parent_dir}/correlation_values.txt', 'a') as f:
    f.write(f"Pearson correlation coefficient: {rho}\n")
    f.write(f"P-value: {p_value}\n")

list_X = np.array(list_otdd).reshape(-1, 1)
list_y = np.array(list_nm)
model = LinearRegression().fit(list_X, list_y)
list_y_pred = model.predict(list_X)

plt.figure(figsize=(10, 8))
# sns.regplot(x=x, y=y, ci=95, scatter_kws={'s': 100}, line_kws={'color': 'blue'})
plt.scatter(list_otdd, list_nm, s=100, color='blue', label='Data points')
plt.plot(list_otdd, list_y_pred, color='red', linewidth=2, label='Fitted line')

plt.xlabel('OTDD')
plt.ylabel('NEW DISTANCE')
plt.title(f'Distance Correlation: *NIST Datasets {rho:.4f}, {p_value:.4f}')

plt.legend()
plt.savefig(f'{parent_dir}/distance_correlation.png')

