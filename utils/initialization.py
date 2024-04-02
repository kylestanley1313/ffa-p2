import numpy as np
import torch
from sklearn.decomposition import PCA
from typing import Generator

from utils.utils import gen_seeds



class PCALoadingInitializer(object):

    def __init__(
            self, 
            method: str, 
            num_facs: int, 
            init_prop: float, 
            generator: torch.Generator
        ) -> None:
        self.method = method
        self.num_facs = num_facs
        self.init_prop = init_prop
        self.gen = generator

    def __call__(self, dataloader: Generator) -> torch.Tensor:
        
        # Read in (possibly subsampled) data
        data = []
        n = 0
        for batch in dataloader:
            n_batch = len(batch)
            num_to_keep = int(n_batch * self.init_prop)
            idx = torch.randperm(n_batch, generator=self.gen)
            data_ = batch[idx[:num_to_keep]]
            data.append(data_)
            n += num_to_keep
        data = torch.cat(data)

        # Prepare PCA estimator
        seed = gen_seeds(self.gen, 1)
        pca = PCA(self.num_facs, svd_solver=self.method, random_state=seed)

        # Initialize loadings then write to file
        # NOTE: If X = USV^T, then covariance is
        #           C = 1/(n-1) X^TX = V(S^2 / (n-1)) V^T
        pca.fit(data)
        loads = np.matmul(
            np.diag(pca.singular_values_) / np.sqrt(n - 1),
            pca.components_
        )
        loads = torch.tensor(loads, dtype=torch.float64)

        return loads