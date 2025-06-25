import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t
from typing import Iterable, List, Optional

from utils import gen_tensors


__all__ = [

    # Datsets
    'CentralizedCovarianceDataset',
    'DistributedCovarianceDataset',
    'DistributedStratifiedCovarianceDataset',

    # Samplers
    'DistributedDatasetSampler',
    'DistributedStratifiedDatasetBatchSampler',
    'DistributedStratifiedDatasetSampler',

    # DataLoaders
    'BasicDataLoader',
    'StratifiedDataLoader',
    'StratifiedTorchDataLoader',

]



# -------------------- DATASETS -------------------- # 


# TODO: 
#   - Update estimation scripts
#   - Update compute_inv_err_cov
#   - Update run_simulations

class CentralizedCovarianceDataset(Dataset):

    def __init__(self, dir_cov: str, dir_idx: str, split: str) -> None:

        idx = torch.load(os.path.join(dir_idx, f'idx.pt'))
        points_loader = gen_tensors(dir_cov, 'points-offband', sort_by=('i', int))
        cov_loader = gen_tensors(dir_cov, f'cov-offband_split-{split}', sort_by=('i', int))
        points_list = []
        cov_list = []
        start = 0
        for points, cov in zip(points_loader, cov_loader):
            sz = len(cov)
            mask = torch.logical_and(idx >= start, idx < start + sz)
            idx_ = idx[mask] - start
            points_list.append(points[idx_])
            cov_list.append(cov[idx_])
            start += sz

        self.points = torch.cat(points_list)
        self.cov = torch.cat(cov_list)

    def __len__(self):
        return len(self.cov)
    
    def __getitem__(self, index):
        return self.points[index], self.cov[index]
    
    def storage(self):
        """Returns size of dataset (in bytes)."""
        points_sz = sys.getsizeof(self.points.untyped_storage())
        cov_sz = sys.getsizeof(self.cov.untyped_storage())
        return points_sz + cov_sz
    

class DistributedCovarianceDataset(Dataset):

    def __init__(
            self, 
            dir_cov: str, 
            dir_idx: str,
            split: str,
            rank: int, 
            world_size: int
        ) -> None:

        # Get point counts for all ranks and load points/cov for this rank
        self.rank_counts = {}
        for r in range(world_size): 

            idx = torch.load(os.path.join(dir_idx, f'idx-{rank}.pt'))
            self.rank_counts[r] = len(idx)
            
            if rank == r: 
                
                points_loader = gen_tensors(dir_cov, 'points', sort_by=('i', int))
                cov_loader = gen_tensors(dir_cov, f'cov_split-{split}', sort_by=('i', int))
                points_list = []
                cov_list = []
                start = 0
                for points, cov in zip(points_loader, cov_loader):
                    sz = len(cov)
                    mask = torch.logical_and(idx >= start, idx < start + sz)
                    idx_ = idx[mask] - start
                    points_list.append(points[idx_])
                    cov_list.append(cov[idx_])
                    start += sz

        self.points = torch.cat(points_list)
        self.cov = torch.cat(cov_list)

    def __len__(self):
        return len(self.cov)

    def __getitem__(self, index):
        return self.points[index], self.cov[index]
    
    def storage(self):
        """Returns size of dataset (in bytes)."""
        points_sz = sys.getsizeof(self.points.untyped_storage())
        cov_sz = sys.getsizeof(self.cov.untyped_storage())
        return points_sz + cov_sz


class DistributedStratifiedCovarianceDataset(Dataset):

    def __init__(
            self, 
            dir_cov: str,
            dir_idx: str, 
            split: str,
            fold: Optional[int],
            rank: int, 
            world_size: int, 
        ) -> None:

        # Read in this rank's indices and strata
        idx = torch.load(os.path.join(dir_idx, f'idx-{rank}.pt'))
        strat = torch.load(os.path.join(dir_idx, f'strat-{rank}.pt'))
        
        # Get points and covariance loaders
        points_loader = gen_tensors(dir_cov, 'points_', sort_by=('i', int))
        prefix = f'cov_split-{split}_'
        if split != 'full': 
            prefix += f'v-{fold}_'
        cov_loader = gen_tensors(dir_cov, prefix, sort_by=('i', int))

        # Get this rank's points, strata, and covariance
        start = 0
        points_list = []
        strat_list = []
        cov_list = []
        for points, cov  in zip(points_loader, cov_loader):
            sz = len(cov)
            mask = torch.logical_and(idx >= start, idx < start + sz)
            idx_ = idx[mask]
            strat_list.append(strat[mask])
            points_list.append(points[idx_ - start])
            cov_list.append(cov[idx_ - start])
            start += sz 
        points = torch.cat(points_list)
        strat = torch.cat(strat_list)
        cov = torch.cat(cov_list)

        # Create dictionaries that map stratum to points/cov
        num_strata = 2 * world_size + 1
        self.strat_points = {}
        self.strat_cov = {}
        for s in range(num_strata):
            mask = strat == s
            self.strat_points[s] = points[mask]
            self.strat_cov[s] = cov[mask]

    def __len__(self):
        return len(self.cov)

    def __getitem__(self, index):
        return self.points[index], self.cov[index]
    
    def set_stratum(self, stratum):
        self.points = self.strat_points[stratum]
        self.cov = self.strat_cov[stratum]

    def set_full_dataset(self):
        self.points = torch.cat(list(self.strat_points.values()))
        self.cov = torch.cat(list(self.strat_cov.values()))
    
    def storage(self):
        """Returns size of dataset (in bytes)."""
        self.set_full_dataset()
        points_sz = sys.getsizeof(self.points.untyped_storage())
        cov_sz = sys.getsizeof(self.cov.untyped_storage())
        return points_sz + cov_sz
   


# -------------------- SAMPLERS -------------------- #

class DistributedDatasetSampler(Sampler):

    def __init__(self, dataset, gen):
        self.dataset = dataset
        self.gen = gen
        self.n_iters = max(dataset.rank_counts.values())

    def __iter__(self):
        idx = torch.randperm(len(self.dataset), generator=self.gen)
        pad_size = self.n_iters - len(idx)
        idx_pad = torch.randperm(len(idx), generator=self.gen)[:pad_size]
        return iter(idx.tolist() + idx_pad.tolist())


class DistributedStratifiedDatasetSampler(Sampler):

    def __init__(
            self, 
            dataset: DistributedStratifiedCovarianceDataset, 
            gen: torch.Generator
        ) -> None:
        self.dataset = dataset
        self.gen = gen

    def __iter__(self):
        idx = torch.randperm(len(self.dataset), generator=self.gen)
        return iter(idx.tolist())
    

class DistributedStratifiedDatasetBatchSampler(object):

    def __init__(
            self, 
            dataset: DistributedStratifiedCovarianceDataset, 
            batch_size: int,
            gen: torch.Generator
        ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.gen = gen
        self.drop_last = False

    def __iter__(self):
        idx = torch.randperm(len(self.dataset), generator=self.gen)
        for idx_batch in torch.split(idx, self.batch_size):
            yield idx_batch.tolist()

    def __len__(self):
        n = len(self.dataset)
        quotient = n // self.batch_size
        remainder = n % self.batch_size
        if remainder:
            return quotient
        else: 
            return quotient + 1
        


# -------------------- DATALOADERS -------------------- #
        
class BasicDataLoader(object):

    def __init__(self, dataset, sampler, batch_size):
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size

    def __iter__(self):
        curr_batch = torch.zeros(self.batch_size, dtype=torch.int32)
        cnt = 0
        for idx in self.sampler:
            curr_batch[cnt] = idx
            cnt += 1
            if cnt == self.batch_size:
                yield self.dataset[curr_batch]
                curr_batch = torch.zeros(self.batch_size, dtype=torch.int32)
                cnt = 0

        # Yield the last batch if it's not a complete batch
        if cnt > 0:
            yield self.dataset[curr_batch[:cnt]]
            
        
class StratifiedDataLoader(object):

    def __init__(
            self, 
            dataset: DistributedStratifiedCovarianceDataset, 
            sampler: Optional[DistributedStratifiedDatasetSampler] = None, 
            batch_sampler: Optional[DistributedStratifiedDatasetBatchSampler] = None,
            batch_size: Optional[int] = None
        ) -> None:
        self.dataset = dataset

        if (sampler is None) == (batch_sampler is None):
            raise ValueError("Exactly one of `sampler` and `batch_size` must be None.") 
        self.sampler = sampler
        self.batch_sampler = batch_sampler
    
        if sampler: 
            self.batch_size = batch_size if batch_size else 1
        else: 
            if batch_size is not None:
                raise ValueError("`batch_sampler` and `batch_size` are mutually exclusive.")
        
        self.iterator = self._get_iterator()

    def __iter__(self):
        return self.iterator()

    def __len__(self):
        return len(self.dataset)

    def set_stratum(self, stratum):
        self.dataset.set_stratum(stratum)

    def set_full_dataset(self):
        self.dataset.set_full_dataset()

    def _get_iterator(self):
        if self.sampler:
            return self._iterator_for_sampler
        else:
            return self._iterator_for_batched_sampler

    def _iterator_for_sampler(self):
        idx = torch.tensor(list(self.sampler), dtype=torch.int32)
        for idx_batch in torch.split(idx, self.batch_size):
            yield self.dataset[idx_batch]

    def _iterator_for_batched_sampler(self):
        for idx_batch in self.batch_sampler:
            yield self.dataset[idx_batch]


class StratifiedTorchDataLoader(DataLoader):

    def __init__(
            self, 
            dataset: Dataset, 
            batch_size: int | None = 1, 
            shuffle: bool | None = None, 
            sampler: Sampler | Iterable | None = None, 
            batch_sampler: Sampler[List] | Iterable[List] | None = None, 
            num_workers: int = 0, 
            collate_fn: _collate_fn_t | None = None, 
            pin_memory: bool = False, 
            drop_last: bool = False, 
            timeout: float = 0, 
            worker_init_fn: _worker_init_fn_t | None = None, 
            multiprocessing_context=None, 
            generator=None, 
            *, 
            prefetch_factor: int | None = None, 
            persistent_workers: bool = False, 
            pin_memory_device: str = ""
        ):
        super().__init__(
            dataset, 
            batch_size, 
            shuffle, 
            sampler, 
            batch_sampler, 
            num_workers, 
            collate_fn, 
            pin_memory, 
            drop_last, 
            timeout, 
            worker_init_fn, 
            multiprocessing_context, 
            generator, 
            prefetch_factor=prefetch_factor, 
            persistent_workers=persistent_workers, 
            pin_memory_device=pin_memory_device
        )

    def set_stratum(self, stratum):
        self.dataset.set_stratum(stratum)

    def set_full_dataset(self):
        self.dataset.set_full_dataset()
