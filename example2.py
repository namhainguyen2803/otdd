from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.method import NewDatasetDistance

# Load data
loaders_src  = load_torchvision_data('KMNIST', valid_size=0, resize = 28, maxsize=None)[0]
loaders_tgt  = load_torchvision_data('USPS',  valid_size=0, resize = 28, maxsize=None, datadir="data/USPS")[0]





# Instantiate distance
# dist = NewDatasetDistance(loaders_src['train'], 
#                           loaders_tgt['train'],
#                           p=2,
#                           device='cpu')

# d = dist.distance(maxsamples=None, num_projection=5000)
# print(f'OTDD(MNIST,FashionMNIST)={d:8.2f}')

# print(len(loaders_src['train']), len(loaders_tgt['train']))

# for (x, y) in loaders_src['train']:
#     print(x.shape, y.shape)