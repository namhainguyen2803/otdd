from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.distance import DatasetDistance

# Load data
loaders_src  = load_torchvision_data('MNIST', valid_size=0, resize = 28, maxsize=20000)[0]
loaders_tgt  = load_torchvision_data('FashionMNIST',  valid_size=0, resize = 28, maxsize=20000)[0]

# Instantiate distance
dist = DatasetDistance(loaders_src['train'], loaders_tgt['train'],
                          inner_ot_method = 'exact',
                          debiased_loss = True,
                          p = 2, entreg = 1e-1,
                          device='cpu')

d = dist.distance(maxsamples = 20000)
print(f'OTDD(MNIST,FashionMNIST)={d:8.2f}')

print(len(loaders_src['train']), len(loaders_tgt['train']))

# for (x, y) in loaders_src['train']:
#     print(x.shape, y.shape)