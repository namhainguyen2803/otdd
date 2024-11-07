"Proposed method"

import logging
logger = logging.getLogger(__name__)


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

from .utils import load_full_dataset, extract_data_targets, process_device_arg, generate_uniform_unit_sphere_projections
from .wasserstein import Sliced_Wasserstein_Distance, Wasserstein_One_Dimension


class NewDatasetDistance():
    """The main class for the new proposed method

    Arguments:
        D1 (Dataset or Dataloader): the first (aka source) dataset.
        D2 (Dataset or Dataloader): the second (aka target) dataset.
    """
    def __init__(self, 
                D1, 
                D2,
                src_embedding=None,
                tgt_embedding=None,
                min_labelcount=2,
                a=None,
                b=None,
                p=2,
                device="cpu",
                precision="float"):

        self.D1 = D1
        self.D2 = D2

        self.p = p 
        self.device = device

        self.a = None
        self.b = None

        self.X1, self.X2 = None, None
        self.Y1, self.Y2 = None, None
        self.dict_data_1, self.dict_data_2 = None, None

        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding

        self.min_labelcount = min_labelcount

        self.precision = precision

        if self.src_embedding is not None or self.tgt_embedding is not None:
            self.feature_cost = partial(FeatureCost,
                                   src_emb = self.src_embedding,
                                   src_dim = (3,28,28),
                                   tgt_emb = self.tgt_embedding,
                                   tgt_dim = (3,28,28),
                                   p = self.p, device=self.device)

        self.src_embedding = None
        self.tgt_embedding = None

        self._init_data(self.D1, self.D2)

    def _init_data(self, D1, D2):
        """ Preprocessing of datasets. Extracts value and coding for effective
        (i.e., those actually present in sampled data) class labels.
        """

        targets1, classes1, idxs1 = extract_data_targets(D1)
        targets2, classes2, idxs2 = extract_data_targets(D2)

        ## Get effective dataset number of samples
        self.idxs1, self.idxs2 = idxs1, idxs2
        self.n1 = len(self.idxs1)
        self.n2 = len(self.idxs2)

        self.targets1 = targets1
        vals1, cts1 = torch.unique(targets1[idxs1], return_counts=True)

        ## Ignore everything with a label occurring less than k times
        self.V1 = torch.sort(vals1[cts1 >= self.min_labelcount])[0] # return all class_ids that pass the requirement

        self.targets2 = targets2
        vals2, cts2 = torch.unique(targets2[idxs2], return_counts=True)
        self.V2 = torch.sort(vals2[cts2 >= self.min_labelcount])[0]

        self.classes1 = [classes1[i] for i in self.V1]
        self.classes2 = [classes2[i] for i in self.V2]

        ## Keep track of real classes vs indices (always 0 to n)(useful if e.g., missing classes):
        self.class_to_idx_1 = {i: c for i, c in enumerate(self.V1)}
        self.class_to_idx_2 = {i: c for i, c in enumerate(self.V2)}


    def _load_datasets(self, maxsamples=None, device=None):
        """ Dataset loading, wrapper for `load_full_dataset` function.

        Loads full datasets into memory (into gpu if in CUDA mode).

        Arguments:
            maxsamples (int, optional): maximum number of samples to load.
            device (str, optional): if provided, will override class attribute device.
        """
        logger.info('Concatenating feature vectors...')

        ## We probably don't ever want to store the full datasets in GPU
        device = 'cpu'
        dtype = torch.DoubleTensor if self.precision == 'double' else torch.FloatTensor
        reindex_start_d2 = 0
        if self.X1 is None or self.Y1 is None:
            self.X1, self.Y1, self.dict_data_1 = self.load_full_dataset(self.D1, 
                                                                        labels_keep=self.V1,
                                                                        maxsamples=maxsamples,
                                                                        device=device,
                                                                        dtype=dtype,
                                                                        reindex=True,
                                                                        reindex_start=0)

        if self.X2 is None or self.Y2 is None:
            self.X2, self.Y2, self.dict_data_2 = self.load_full_dataset(self.D2, 
                                                                        labels_keep=self.V2,
                                                                        maxsamples=maxsamples,
                                                                        device=device,
                                                                        dtype=dtype,
                                                                        reindex=True,
                                                                        reindex_start=reindex_start_d2)


        logger.info("Full datasets sizes")
        logger.info(" * D1 = {} x {} ({} unique labels)".format(*
                                                          self.X1.shape, len(self.V1)))
        logger.info(" * D2 = {} x {} ({} unique labels)".format(*
                                                          self.X2.shape, len(self.V2)))


    def load_full_dataset(self, data,
                        labels_keep=None,
                        maxsamples=None, 
                        device='cpu', 
                        dtype=torch.FloatTensor,
                        feature_embedding=None,
                        reindex=False, reindex_start=0):
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

            X.append(x.squeeze().view(x.shape[0], -1))
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

    def _compute_moments_projected_distrbution(self, X, projection_matrix, k):
        """
        encode distribution into a vector having length num_moments, 
        which calculates high-order moment of projected distribution having support X.

        projection_matrix has shape  R^(num_projection, dim)
        X has shape R^(num_examples, dim)
        k has shape R^(num_projection)
        """
        num_projection = projection_matrix.shape[0]
        num_examples = X.shape[0]
        X_projection = torch.matmul(X, projection_matrix.t())  # shape == (num_examples, num_projection)

        # moment_X_projection = torch.pow(input=X_projection.permute(1, 0).unsqueeze(-1), exponent=k) 
        # shape == (num_projection, num_examples, num_moments)

        if k.ndim == 1:
            k = k.unsqueeze(-1)
        moment_X_projection = torch.pow(input=X_projection.unsqueeze(1), exponent=k.permute(1, 0))
        moment_X_projection = moment_X_projection.permute(2, 0, 1) 
        # shape == (num_projection, num_examples, num_moments)

        avg_moment_X_projection = torch.sum(moment_X_projection, dim=1) / num_examples # shape == (num_projection, num_moments)
        # normalized_avg_moment_X_projection = torch.pow(input=avg_moment_X_projection, exponent=1/k)
        # print(normalized_avg_moment_X_projection)
        # print(avg_moment_X_projection[:2], normalized_avg_moment_X_projection[:2])
        # print()
        return X_projection, avg_moment_X_projection # shape == (num_examples, num_projection); (num_projection, num_moments)


    def _compute_projected_dataset_matrix(self, dict_data, projection_matrix, projection_matrix_2, k):
        
        proj_matrix_dataset = list()
        for (cls_id, data) in dict_data.items():
            X_projection, avg_moment_X_projection = self._compute_moments_projected_distrbution(X=data, projection_matrix=projection_matrix, k=k) 
            # print(f"cls_id {cls_id}, avg_moment_X_projection: {torch.isnan(avg_moment_X_projection).any()}")
            # shape == (num_examples, num_projection); (num_projection, num_moments)
            X_projection = torch.permute(X_projection, dims=(1, 0)) # shape == (num_projection, num_examples)

            # X_projection.shape == (num_projection, num_examples)
            # avg_moment_X_projection.shape == (num_projection, num_moments)

            num_examples = X_projection.shape[1]
            num_projection = avg_moment_X_projection.shape[0]
            num_moments = avg_moment_X_projection.shape[1]

            # print(X_projection.shape, avg_moment_X_projection.shape, avg_moment_X_projection.unsqueeze(1).expand(num_projection, num_examples, num_moments).shape)
            h = torch.cat((X_projection.unsqueeze(-1),
                            avg_moment_X_projection.unsqueeze(1).expand(num_projection, num_examples, num_moments)), 
                            dim=2) # shape == (num_projection, num_examples, num_moments+1)
            
            proj_matrix_dataset.append(h)
        
        proj_matrix_dataset = torch.cat(proj_matrix_dataset, dim=1) # shape == (num_projection, total_examples, num_moments+1) # total_examples = sum(num_examples)

        proj_proj_matrix_dataset = torch.matmul(proj_matrix_dataset, projection_matrix_2.unsqueeze(-1)).squeeze(-1) # shape == (num_projection, total_examples)

        return proj_proj_matrix_dataset.transpose(1, 0) # shape == (total_examples, num_projection)

    def distance(self, maxsamples=None, num_projection=10000):
        """
        self.X: tensor of features 60000x1x28x28 = 60000x784
        self.Y: tensor of labels corresponding to features to be considered [60000]
        self.V: set of all labels to be considered
        """

        self._load_datasets(maxsamples=maxsamples)

        dtype = torch.DoubleTensor if self.precision == 'double' else torch.FloatTensor

        print(self.X1.shape, self.Y1.shape)
        print(self.X2.shape, self.Y2.shape)

        chunk = 1000
        chunk_num_projection = num_projection // chunk

        num_moments = 3

        all_sw = list()
        for i in range(chunk_num_projection):

            # moments = torch.randint(1, 6, (chunk, num_moments))

            row = torch.tensor([1])
            num_moments = len(row)
            moments = row.unsqueeze(0).repeat(chunk, 1)

            projection_matrix = generate_uniform_unit_sphere_projections(dim=self.X1.shape[1],num_projection=chunk, device=self.device)
            projection_matrix = projection_matrix.type(dtype)

            # use this matrix to project vector concat([projected_x, high-order moment]) into 1D, has shape (chunk, num_moment+1)
            projection_matrix_2 = generate_uniform_unit_sphere_projections(dim=num_moments+1, num_projection=chunk, device=self.device)
            projection_matrix_2 = projection_matrix_2.type(dtype)

            proj_proj_matrix_dataset_1 = self._compute_projected_dataset_matrix(dict_data=self.dict_data_1,
                                                                                projection_matrix=projection_matrix,
                                                                                projection_matrix_2=projection_matrix_2,
                                                                                k=moments) # shape == (total_examples_of_dataset_1, num_projection)

            proj_proj_matrix_dataset_2 = self._compute_projected_dataset_matrix(dict_data=self.dict_data_2,
                                                                                projection_matrix=projection_matrix,
                                                                                projection_matrix_2=projection_matrix_2,
                                                                                k=moments) # shape == (total_examples_of_dataset_2, num_projection)

            num_supports_source = self.X1.shape[0]
            num_supports_target = self.X2.shape[0]

            w_1d = Wasserstein_One_Dimension(X=proj_proj_matrix_dataset_1,
                                            Y=proj_proj_matrix_dataset_2,
                                            p=self.p,
                                            device=self.device)  # shape (chunk)

            del proj_proj_matrix_dataset_1
            del proj_proj_matrix_dataset_2

            # contains_nan = torch.isnan(w_1d).any()
            # print(f"w_1d {contains_nan}")

            if self.p != 1:
                sw = torch.pow(input=w_1d, exponent=self.p)
            else:
                sw = w_1d
            all_sw.append(sw)

            # print(f"cac {torch.mean(sw)}")
            a = torch.pow(torch.mean(sw), exponent=1/self.p)
            print(i, sw.shape, a)
            # print(i, sw.shape)

        all_sw = torch.cat(all_sw, dim=0)
        print(f"Cac: {all_sw.shape}")
        assert all_sw.shape[0] == chunk_num_projection * chunk

        return torch.pow(torch.mean(all_sw), exponent=1/self.p)     


    def distance_without_labels(self, maxsamples, num_projection):
        self._load_datasets(maxsamples=maxsamples)

        print(self.X1.shape, self.Y1.shape)
        print(self.X2.shape, self.Y2.shape)

        chunk = 1000
        chunk_num_projection = num_projection // chunk

        all_sw = list()
        for i in range(chunk_num_projection):
            sw = Sliced_Wasserstein_Distance(X=self.X1, Y=self.X2, num_projection=chunk)
            all_sw.append(sw)
        all_sw = torch.cat(all_sw, dim=0)
        print(f"Cac: {all_sw.shape}")
        assert all_sw.shape[0] == chunk_num_projection * chunk

        return torch.pow(torch.mean(all_sw), exponent=1/self.p)
