"Proposed method"

import logging
logger = logging.getLogger(__name__)
import math

import ot
try:
    import ot.gpu
except:
    logger.warning('ot.gpu not found - coupling computation will be in cpu')

import torch
import torch.utils.data.dataloader as dataloader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm.autonotebook import tqdm
import numpy as np
import random

from .utils import extract_data_targets, process_device_arg, generate_uniform_unit_sphere_projections, normalizing_moments_2, generate_Gaussian_projectors
from .wasserstein import Sliced_Wasserstein_Distance, Wasserstein_One_Dimension
from otdd.pytorch.moments import compute_label_stats

import time

from otdd.pytorch.utils import generate_and_plot_data

import geoopt



def load_full_dataset(data, labels_keep=None, maxsamples=None, device='cpu', precision="float", feature_embedding=None, reindex=False, reindex_start=0):
    """ Loads full dataset into memory.

    Arguments:
        targets (bool, or 'infer'): Whether to collect and return targets (labels) too
        labels_keep (list): If provided, will only keep examples with these labels
        reindex (bool): Whether/how to reindex labels. If True, will
                            reindex to {reindex_start,...,reindex_start+num_unique_labels}.
        maxsamples (int): Maximum number of examples to load. (this might not equal
                        actual size of return tensors, if label_keep also provided)

    Returns:
        X (tensor): tensor of dataset features, stacked along first dimension
        Y (tensor): tensor of dataset targets

    """
    dtype = torch.DoubleTensor if precision == 'double' else torch.FloatTensor
    device = process_device_arg(device)
    orig_idxs = None
    if type(data) == dataloader.DataLoader:
        loader = data
        if maxsamples:
            if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'indices'): # no vao day
                if len(loader.sampler.indices) <= maxsamples:
                    logger.warning('Maxsamples is greater than number of effective examples in loader. Will not subsample.')
                else:
                    ## Resample from sampler indices. 
                    orig_idxs = loader.sampler.indices
                    idxs = np.sort(np.random.choice(orig_idxs, maxsamples, replace=False))
                    loader.sampler.indices = idxs # sua index, chi iterate nhung sample index duoc chon
            elif hasattr(loader, 'dataset'): # This probably means the sampler is not a subsampler. So len(dataset) is indeed true size.
                if len(loader.dataset) <= maxsamples:
                    logger.warning('Maxsamples is greater than number of examples in loader. Will not subsample.')
                else:
                    ## Create new sampler
                    idxs = np.sort(np.random.choice(len(loader.dataset), maxsamples, replace=False))
                    sampler = SubsetRandomSampler(idxs)
                    loader = dataloader.DataLoader(data, sampler=sampler, batch_size=batch_size)
            else:
                ## I don't think we'll ever be in this case.
                print('Warning: maxsamplers provided but loader doesnt have subsampler or dataset. Cannot subsample.')  
    X = []
    Y = []
    seen_targets = {}
    keeps = None
    for batch in tqdm(loader, leave=False):
        x = batch[0]
        if (len(batch) == 2):
            y = batch[1]
        if feature_embedding is not None:
            try:
                x = feature_embedding(x.type(dtype).cuda()).detach().to(device)
            except:
                x = feature_embedding(x.type(dtype).to(device)).detach()
        else:
            x = x.type(dtype).to(device)
        X.append(x.to(device))
        Y.append(y.to(device).squeeze())
    X = torch.cat(X)
    Y = torch.cat(Y)
    if labels_keep is not None: # Filter out examples with unwanted label
        keeps = np.isin(Y.cpu(), labels_keep)
        X = X[keeps,:] 
        Y = Y[keeps]
    if orig_idxs is not None: # sua lai index ve dung vi tri ban dau
        loader.sampler.indices = orig_idxs
    if reindex:
        labels = sorted(torch.unique(Y).tolist())
        reindex_vals = range(reindex_start, reindex_start + len(labels))
        lmap = dict(zip(labels, reindex_vals))
        Y = torch.LongTensor([lmap[y.item()] for y in Y]).to(device)
    dict_data = dict()
    for cls in torch.unique(Y):
        dict_data[cls] = X[Y == cls]
    return X, Y, dict_data


class Embeddings_sOTDD():

    def __init__(self, p=2, device="cpu"):
        self.p = p 
        self.device = device
    

    def _project_data(self, X, projection_matrix):
        """
        project X which has shape R^(c, h, w) to a number
        projection_matrix can have shape (num_projections, flatten_dim)
        """
        if X.ndim != 1:
            X = X.reshape(X.shape[0], -1)
        X_projection = torch.matmul(X, projection_matrix.t())  # shape == (num_examples, num_projection)
        return X_projection
    
    def _project_distribution(self, mean, cov, w, theta, A):
        """
        mean has shape R^(num_cls, flatten_dim)
        cov has shape R^(num_cls, flatten_dim, flatten_dim)
        w has shape R^(L, 2)
        theta has shape R^(L, flatten_dim)
        A has shape R^(L, flatten_dim, flatten_dim)
        """
        num_classes = cov.shape[0]
        flatten_dim = cov.shape[1]
        num_projections = w.shape[0]

        # projected_mean = torch.matmul(mean, theta.transpose(0, 1))
        # log_cov = geoopt.linalg.sym_logm(cov)
        # projected_cov = torch.sum(torch.matmul(A, log_cov.unsqueeze(0))[:, torch.arange(flatten_dim), torch.arange(flatten_dim)], dim=1)
        # return w[:, 0] * projected_mean + w[:, 1] * projected_cov

        projected_mean = torch.matmul(mean, theta.transpose(0, 1)) # shape == ()
        log_cov = geoopt.linalg.sym_logm(cov)
        projected_cov = (A.unsqueeze(0) * log_cov.unsqueeze(1)).reshape(num_classes, num_projections, -1).sum(-1)
        projected_distributions = torch.stack([projected_mean, projected_cov], dim=-1)
        del projected_cov
        del projected_mean
        del log_cov
        avg_projected_distributions = torch.sum(projected_distributions * w, dim=-1)
        return avg_projected_distributions


    def get_embeddings(self, dict_data, cls_stats, theta, psi):

        proj_matrix_dataset = list()

        M, C = cls_stats[0], cls_stats[1]
        projected_distribution = self._project_distribution(mean=M, cov=C, w=theta[0], theta=theta[1], A=theta[2]) # shape == (num_cls, num_projections)

        del cls_stats

        for (cls_id, data) in dict_data.items():

            # compute projected distribution and projected data
            projected_cls_data = self._project_data(X=data, projection_matrix=theta[1]) # shape == (num_examples, num_projections)

            # concat projected distribution into projected data
            projected_cls_data = projected_cls_data.T # shape == (num_projections, num_examples)
            projected_cls_distribution = projected_distribution[cls_id].unsqueeze(-1)

            num_projection, num_examples = projected_cls_data.shape[0], projected_cls_data.shape[1]

            h = torch.cat((projected_cls_data.unsqueeze(-1),
                            projected_cls_distribution.unsqueeze(1).expand(num_projection, num_examples, 1)), 
                            dim=2) # shape == (num_projection, num_examples, num_moments+1)

            # has_nan = torch.isnan(h).any()
            # print(f"If matrix h has nan: {has_nan}")

            proj_matrix_dataset.append(h)
        
        proj_matrix_dataset = torch.cat(proj_matrix_dataset, dim=1) # shape == (num_projection, total_examples, num_moments+1)
        proj_proj_matrix_dataset = torch.matmul(proj_matrix_dataset, psi.unsqueeze(-1)).squeeze(-1) # shape == (num_projection, total_examples)
        return proj_proj_matrix_dataset.transpose(1, 0) # shape == (total_examples, num_projection)


def compute_pairwise_distance(list_dict_data=None, list_stats_data=None, device='cpu', num_projections=10000, evaluate_time=False, **kwargs):
    dimension = kwargs.get('dimension', 768)
    num_channels = kwargs.get('num_channels', 1)
    precision = kwargs.get('precision', "float")
    p = kwargs.get('p', 2)
    chunk = kwargs.get('chunk', 1000)

    if num_channels != 1:
        flatten_dimension = num_channels * dimension * dimension
    else:
        flatten_dimension = num_channels * dimension

    if num_projections < chunk:
        chunk = num_projections
        chunk_num_projection = 1
    else:
        chunk_num_projection = num_projections // chunk
    dtype = torch.DoubleTensor if precision == 'double' else torch.FloatTensor

    list_theta = list()
    list_psi = list()

    for i in range(chunk_num_projection):

        chunk_theta, chunk_A = generate_Gaussian_projectors(dim=flatten_dimension, num_projection=chunk, device=device, dtype=dtype)
        chunk_psi = generate_uniform_unit_sphere_projections(dim=2, num_projection=chunk, device=device, dtype=dtype)
        chunk_w = generate_uniform_unit_sphere_projections(dim=2, num_projection=chunk, device=device, dtype=dtype)

        list_theta.append([chunk_w, chunk_theta, chunk_A])
        list_psi.append(chunk_psi)
    

    embeddings = Embeddings_sOTDD(device=device)

    list_w1d = list()

    if evaluate_time is True:
        start = time.time()
    for projection_chunk_id in range(chunk_num_projection):
        list_chunk_embeddings = list()
        for dataset_id in range(len(list_dict_data)):
            chunk_dataset_embeddings = embeddings.get_embeddings(dict_data=list_dict_data[dataset_id],
                                                                cls_stats=list_stats_data[dataset_id],
                                                                theta=list_theta[projection_chunk_id], 
                                                                psi=list_psi[projection_chunk_id])
            list_chunk_embeddings.append(chunk_dataset_embeddings)

        list_chunk_w1d = list()
        for i in range(len(list_chunk_embeddings)):
            for j in range(i+1, len(list_chunk_embeddings)):
                w_1d = Wasserstein_One_Dimension(X=list_chunk_embeddings[i],
                                                Y=list_chunk_embeddings[j],
                                                p=p,
                                                device=device)  # shape (chunk)
                list_chunk_w1d.append(w_1d.reshape(-1, 1))
        list_chunk_w1d = torch.cat(list_chunk_w1d, dim=1)
        list_w1d.append(list_chunk_w1d)


        cac = torch.cat(list_w1d, dim=0)
        if p != 1:
            sw = torch.pow(input=cac, exponent=p)
        else:
            sw = cac
        sw = torch.pow(torch.mean(sw, dim=0), exponent=1/p) 
        
        print(f"sw: {sw}")
        

    list_w1d = torch.cat(list_w1d, dim=0)
    if p != 1:
        sw = torch.pow(input=list_w1d, exponent=p)
    else:
        sw = list_w1d
    sw = torch.pow(torch.mean(sw, dim=0), exponent=1/p) 

    if evaluate_time is True:
        end = time.time()
        sotdd_time_taken = end - start
        return sw, sotdd_time_taken
    else:
        return sw

