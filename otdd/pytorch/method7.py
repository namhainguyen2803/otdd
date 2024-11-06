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

from .utils import load_full_dataset, extract_data_targets, process_device_arg, generate_uniform_unit_sphere_projections, generate_unit_convolution_projections, generate_uniform_unit_sphere_projections_2, generate_moments, normalizing_moments, normalizing_moments_2
from .wasserstein import Sliced_Wasserstein_Distance, Wasserstein_One_Dimension

import time


class Embeddings_sOTDD():

    def __init__(self, 
                list_moments,
                list_theta,
                list_psi,
                list_factorial_moments,
                min_labelcount=2,
                p=2,
                device="cpu",
                precision="float"):

        self.list_moments = list_moments
        self.list_theta = list_theta
        self.list_psi = list_psi
        self.list_factorial_moments = list_factorial_moments
        self.p = p 
        self.device = device
        self.min_labelcount = min_labelcount
        self.precision = precision


    def _init_data(self, D):
        """ Preprocessing of datasets. Extracts value and coding for effective
        (i.e., those actually present in sampled data) class labels.
        """
        targets, classes, idxs = extract_data_targets(D)
        vals, cts = torch.unique(targets[idxs], return_counts=True)
        labels_kept = torch.sort(vals[cts >= self.min_labelcount])[0]
        classes = [classes[i] for i in labels_kept]
        # class_to_idx = {i: c for i, c in enumerate(labels_kept)}
        return classes, labels_kept


    def _load_datasets(self, D, labels_kept=None, maxsamples=None, device='cpu'):
        logger.info('Concatenating feature vectors...')
        ## We probably don't ever want to store the full datasets in GPU
        device = 'cpu'
        dtype = torch.DoubleTensor if self.precision == 'double' else torch.FloatTensor
        X, Y, dict_data = self.load_full_dataset(D, 
                                                labels_keep=labels_kept,
                                                maxsamples=maxsamples,
                                                device=device,
                                                dtype=dtype,
                                                reindex=True,
                                                reindex_start=0)
        return X, Y, dict_data


    def load_full_dataset(self, 
                        data,
                        labels_keep=None,
                        maxsamples=None, 
                        device='cpu', 
                        dtype=torch.FloatTensor,
                        feature_embedding=None,
                        reindex=False, 
                        reindex_start=0):
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
                ## if embedding is cuda, and device='cpu', want to map to device *after*
                ## embedding, to take advantage of CUDA forward pass.
                try:
                    x = feature_embedding(x.type(dtype).cuda()).detach().to(device)
                except:
                    x = feature_embedding(x.type(dtype).to(device)).detach()
            else:
                x = x.type(dtype).to(device)

            # X.append(x.squeeze().view(x.shape[0], -1))
            X.append(x)
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
    

    def _project_X(self, X, projection_matrix, use_conv=False):
        """
        project X which has shape R^(c, h, w) to a number
        projection_matrix can have 
        """
        if use_conv:
            for conv in projection_matrix:
                X = conv(X).detach()
            return X.squeeze(-1).squeeze(-1)
        else:
            X_projection = torch.matmul(X, projection_matrix.t())  # shape == (num_examples, num_projection)
            return X_projection


    def _compute_moments_projected_distrbution(self, X_projection, k, factorial_k):
        """
        encode distribution into a vector having length num_moments, 
        which calculates high-order moment of projected distribution having support X.

        X_projection has shape R^(num_examples, num_projection)
        k has shape R^(num_projection)
        """

        # moment_X_projection = torch.pow(input=X_projection.permute(1, 0).unsqueeze(-1), exponent=k) 
        # shape == (num_projection, num_examples, num_moments)

        if k.ndim == 1:
            k = k.unsqueeze(-1)
        moment_X_projection = torch.pow(input=X_projection.unsqueeze(1), exponent=k.permute(1, 0))
        moment_X_projection = moment_X_projection.permute(2, 0, 1) 
        # shape == (num_projection, num_examples, num_moments)

        avg_moment_X_projection = torch.sum(moment_X_projection, dim=1) / X_projection.shape[0] # shape == (num_projection, num_moments)
        avg_moment_X_projection = normalizing_moments_2(avg_moment_X_projection, k, factorial_k)

        return avg_moment_X_projection # shape == (num_projection, num_moments)


    def _compute_projected_dataset_matrix(self, dict_data, projection_matrix, projection_matrix_2, k, factorial_k, use_conv=False):
        
        proj_matrix_dataset = list()
        for (cls_id, data) in dict_data.items():

            if use_conv is False:
                data = data.reshape(data.shape[0], -1)

            X_projection = self._project_X(X=data, projection_matrix=projection_matrix, use_conv=use_conv) # shape == (num_examples, num_projection)
            avg_moment_X_projection = self._compute_moments_projected_distrbution(X_projection=X_projection, k=k, factorial_k=factorial_k)
            # shape == (num_projection, num_moments)
            X_projection = torch.permute(X_projection, dims=(1, 0)) # shape == (num_projection, num_examples)

            num_examples = X_projection.shape[1]
            num_projection = avg_moment_X_projection.shape[0]
            num_moments = avg_moment_X_projection.shape[1]

            h = torch.cat((X_projection.unsqueeze(-1),
                            avg_moment_X_projection.unsqueeze(1).expand(num_projection, num_examples, num_moments)), 
                            dim=2) # shape == (num_projection, num_examples, num_moments+1)
            
            proj_matrix_dataset.append(h)
        
        proj_matrix_dataset = torch.cat(proj_matrix_dataset, dim=1) 
        # shape == (num_projection, total_examples, num_moments+1)

        proj_proj_matrix_dataset = torch.matmul(proj_matrix_dataset, projection_matrix_2.unsqueeze(-1)).squeeze(-1) # shape == (num_projection, total_examples)

        return proj_proj_matrix_dataset.transpose(1, 0) # shape == (total_examples, num_projection)

    
    def get_embeddings(self, dict_data, maxsamples, num_projections=1000, chunk=100, use_conv=False):

        chunk_num_projection = num_projections // chunk

        dataset_embeddings = list()
        for i in range(chunk_num_projection):
            chunk_dataset_embeddings = self._compute_projected_dataset_matrix(dict_data=dict_data,
                                                                            projection_matrix=self.list_theta[i],
                                                                            projection_matrix_2=self.list_psi[i],
                                                                            k=self.list_moments[i],
                                                                            factorial_k=self.list_factorial_moments[i],
                                                                            use_conv=use_conv) 
            dataset_embeddings.append(chunk_dataset_embeddings)
        
        if chunk_num_projection != 1:
            dataset_embeddings = torch.cat(dataset_embeddings, dim=0)
            return dataset_embeddings
        else:
            return chunk_dataset_embeddings


def compute_pairwise_distance(list_D, device='cpu', num_projections=10000, evaluate_time=False):

    num_moments = 5

    dimension = 32
    num_channels = 3
    use_conv = True
    precision = "float"
    p = 2

    chunk = 100
    chunk_num_projection = num_projections // chunk

    dtype = torch.DoubleTensor if precision == 'double' else torch.FloatTensor


    list_moments = list()
    list_factorial_moments = list()
    list_theta = list()
    list_psi = list()
    for i in range(chunk_num_projection):
        chunk_moments = torch.stack([generate_moments(num_moments=num_moments, min_moment=1, max_moment=None, gen_type="poisson") for lz in range(chunk)])

        unique_chunk_moments = torch.unique(chunk_moments)

        lookup_factorial = list()
        for i in range(len(unique_chunk_moments)):
            lookup_factorial.append(math.factorial(int(unique_chunk_moments[i])))

        factorial_chunk_moments = torch.zeros_like(chunk_moments)
        for i in range(len(unique_chunk_moments)):
            factorial_chunk_moments[chunk_moments == unique_chunk_moments[i]] = lookup_factorial[i]

        if use_conv is True:
            chunk_theta = generate_unit_convolution_projections(image_size=dimension, num_channels=num_channels, num_projection=chunk, device=device, dtype=dtype)
        else:
            chunk_theta = generate_uniform_unit_sphere_projections(dim=dimension, num_projection=chunk, device=device, dtype=dtype)
        
        chunk_psi = generate_uniform_unit_sphere_projections(dim=num_moments+1, num_projection=chunk, device=device, dtype=dtype)

    list_moments.append(chunk_moments)
    list_theta.append(chunk_theta)
    list_psi.append(chunk_psi)
    list_factorial_moments.append(factorial_chunk_moments)

    embeddings = Embeddings_sOTDD(list_moments=list_moments, list_theta=list_theta, list_psi=list_psi, precision=precision, list_factorial_moments=list_factorial_moments, device=device)

    list_dict_data = list()
    for D in list_D:
        X, Y, dict_data = embeddings._load_datasets(D=D, labels_kept=None, maxsamples=None, device=device)
        print(X.shape, Y.shape)
        del X 
        del Y 
        list_dict_data.append(dict_data)

    del list_D

    list_w1d = list()

    if evaluate_time is True:
        all_start_time = time.time()

    duration_periods = dict()

    for ch in range(chunk_num_projection):
        
        list_chunk_embeddings = list()
        for am in range(len(list_dict_data)):
            chunk_dataset_embeddings = embeddings.get_embeddings(dict_data=list_dict_data[am], maxsamples=None, num_projections=chunk, chunk=chunk, use_conv=use_conv)
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
        print(f"cac 1: {list_chunk_w1d.shape}") # 100, 1
        list_w1d.append(list_chunk_w1d)

        if evaluate_time is True:
            period_end_time = time.time()
            duration_periods[(ch + 1) * chunk] = period_end_time - all_start_time

    list_w1d = torch.cat(list_w1d, dim=0)
    print(f"cac 2: {list_w1d.shape}") # 10000, 1
    if p != 1:
        sw = torch.pow(input=list_w1d, exponent=p)
    else:
        sw = list_w1d
    sw = torch.pow(torch.mean(sw, dim=0), exponent=1/p) 
    print(f"cac 3: {sw.shape}")

    if evaluate_time is True:
        # all_end_time = time.time()
        # duration_periods[(ch + 1) * chunk] = all_end_time - all_start_time
        return sw, duration_periods
    else:
        return sw

