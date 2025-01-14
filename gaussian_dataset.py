import torch
import numpy as np
from otdd.pytorch import method_linear_gaussian, method_gaussian
from torch.utils.data import TensorDataset, DataLoader
from otdd.pytorch.distance import DatasetDistance
from scipy import stats
import random


def generate_multivariate_gaussian_dataset(K, M, D):
    """
    Generate a dataset with K labels, each class having M datasets,
    following a multivariate Gaussian distribution of D dimensions.

    Args:
        K (int): Number of labels/classes.
        M (int): Number of samples per class.
        D (int): Number of dimensions.

    Returns:
        data (torch.Tensor): Dataset of shape (K * M, D).
        labels (torch.Tensor): Corresponding labels of shape (K * M,).
    """
    data = []
    labels = []
    means = []
    covariances = []

    data_dict = dict()

    for label in range(K):
        x = torch.randint(1, 100, (1,)).item()
        mean = torch.randn(D) * x
        diag_elements = torch.abs(torch.randn(D))
        cov = torch.diag(diag_elements)
        means.append(mean)
        covariances.append(cov)
        samples = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov).sample((M,))
        data.append(samples)
        labels.extend([label] * M)

        data_dict[label] = samples

    data = torch.cat(data, dim=0)
    labels = torch.tensor(labels)
    means = torch.stack(means)
    covariances = torch.stack(covariances)

    return data, labels, means, covariances, data_dict


if __name__ == "__main__":

    K = 10
    D = 500
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = "cpu"
    num_projection = 10000

    num_datasets = 10
    list_sotdd = list()
    list_otdd = list()
    
    list_dataset_size = random.choices([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], k=10)

    print(f"Selected dataset size: {list_dataset_size}")

    for dataset_size in list_dataset_size:

        data1, labels1, means1, covariances1, data_dict1 = generate_multivariate_gaussian_dataset(K=K, M=dataset_size, D=D)
        data2, labels2, means2, covariances2, data_dict2 = generate_multivariate_gaussian_dataset(K=K, M=dataset_size, D=D)
        dataset1 = TensorDataset(data1, labels1)
        dataloader1 = DataLoader(dataset1, batch_size=16, shuffle=True)
        dataset2 = TensorDataset(data2, labels2)
        dataloader2 = DataLoader(dataset2, batch_size=16, shuffle=True)
        list_dataloader = [dataloader1, dataloader2]
        list_dict_dataset = [data_dict1, data_dict2]
        list_stats = [[means1, covariances1], [means2, covariances2]]
        print("data shape, label shape, mean shape, covariance shape:", data1.shape, labels1.shape, means1.shape, covariances1.shape)
        kwargs = {
            "dimension": D,
            "num_channels": 1,
            "precision": "float",
            "p": 2,
            "chunk": 1000
        }
        sw_dist = method_linear_gaussian.compute_pairwise_distance(list_dict_data=list_dict_dataset, list_stats_data=list_stats, device=DEVICE, num_projections=num_projection, evaluate_time=False, **kwargs)[0]
        dist = DatasetDistance(list_dataloader[0], 
                                list_dataloader[1],
                                inner_ot_method='gaussian_approx',
                                sqrt_method='approximate',
                                nworkers_stats=0,
                                sqrt_niters=20,
                                debiased_loss=True,
                                p = 2, 
                                entreg = 1e-3,
                                device=DEVICE)
        otdd_dist = dist.distance(maxsamples = None).item()

        list_otdd.append(otdd_dist)
        list_sotdd.append(sw_dist)

        
    torch.save({"sotdd": list_sotdd, "otdd": list_otdd}, "dist.pth")
    
    loaded_data = torch.load('dist.pth')
    sotdd_dist = loaded_data['sotdd']
    otdd_dist = loaded_data['otdd']
    pearson_corr, p_value = stats.pearsonr(sotdd_dist, otdd_dist)
    print(len(sotdd_dist), sotdd_dist)
    print(len(otdd_dist), otdd_dist)
    print(pearson_corr, p_value)
