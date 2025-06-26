import csv
import math
import nibabel as nib
import numpy as np
import os
import pandas as pd
import shutil
import subprocess
import sys
import torch
import torch.distributed as dist
import yaml
from configparser import ConfigParser
from collections import namedtuple
from typing import Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union

from utils.model import LowRankCovariance


# -------------------- ERROR CODES -------------------- #

CODE_DIVERGENCE = 51
CODE_NO_CONVERGENCE = 52


# -------------------- CONFIG -------------------- #

def load_config(config_name, path_config=None):
    """Load the specified config section as a namedtuple."""
    # Default location is config.ini in the same directory as this file
    if path_config is None:
        path_config = os.path.join(os.path.dirname(__file__), '..', 'config.ini')

    # Parse the config file
    parser = ConfigParser()
    parser.read(path_config)

    if config_name not in parser:
        raise ValueError(f"Configuration '{config_name}' not found in {path_config}")

    # Convert section to namedtuple
    config_dict = dict(parser[config_name])
    Config = namedtuple('Config', config_dict.keys())
    return Config(**config_dict)


# -------------------- AOMIC -------------------- #

def get_sub_path(sub_num: int) -> str:
    sub_label = str(sub_num).zfill(4)
    sub_path = os.path.join(
        '/storage/home/kms8227/scratch/datasets/ds002785',
        f'derivatives/fmriprep/sub-{sub_label}/func',
        f'sub-{sub_label}_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
    )
    return sub_path


def read_sub_file(sub_num: int) -> torch.Tensor:
    sub_path = get_sub_path(sub_num)
    data = nib.load(sub_path).get_fdata()
    return torch.from_numpy(data).to(torch.float32) 


# -------------------- MISCELLANEOUS -------------------- #

def gen_seeds(gen, size):
    seeds = torch.randint(
        high=torch.iinfo(torch.int32).max, 
        size=(size,), 
        generator=gen, 
        dtype=torch.int32
    )
    if size == 1: 
        return seeds.tolist()[0]
    else: 
        return seeds.tolist()
    

def gen_range(n, batch_size, as_numpy=False):
    start = 0
    while start < n:
        sz = min(batch_size, n - start)
        idx = torch.arange(sz) + start
        if as_numpy: 
            yield idx.numpy()
        else: 
            yield idx
        start += sz
    

def write_generated_tensor(tensor_loader: Generator, dir: str, prefix: str):
    for i, batch in enumerate(tensor_loader):
        path = os.path.join(dir, f'{prefix}_i-{i}_.pt')
        torch.save(batch, path)


def get_field_from_fname(fname, field):
    return fname.split(f'_{field}-')[1].split('_')[0]


def gen_tensors(
        dir: str, 
        prefix: str, 
        batch_size: Optional[int] = None, 
        sort_by: Optional[Tuple[str, type]] = None
    ) -> Generator:
    """Generates batches of a tensor stored across multiple files.  

    Args:
        dir (str): Directory from which to read files.
        prefix (str): File name prefix. 
        batch_size (Optional[int], optional): Optionally load tensor in batches 
            a certain size. Defaults to None.
        sort_by (Optional[Tuple[str, type]], optional): Sort by a field `sort_by[0]`
            having type `sort_by[1]`. Defaults to None.

    Yields:
        Generator: A generator that yields rows of the tensor. 
    """
    
    # Get sorted list of files
    files = [f for f in os.listdir(dir) if f.startswith(prefix) and f.endswith('.pt')]
    if sort_by is not None: 
        fields = [sort_by[1](get_field_from_fname(f, sort_by[0])) for f in files]
        pairs = sorted(zip(files, fields), key=lambda x: x[1])
        files, _ = zip(*pairs)
        files = list(files)
    else: 
        files = sorted(files)

    if batch_size is None:  # each batch is a file
        for f in files:
            yield torch.load(os.path.join(dir, f))

    else: # each batch is of desired size
        leftover = None
        for f in files: 
            array = torch.load(os.path.join(dir, f))
            leftover = array if leftover is None else torch.cat((leftover, array))
            while len(leftover) >= batch_size:
                yield leftover[:batch_size]
                leftover = leftover[batch_size:]
        if leftover is not None and len(leftover) > 0: 
            yield leftover
                

def gen_arrays(
        dir: str, 
        prefix: str, 
        batch_size: Optional[int] = None, 
        sort_by: Optional[Tuple[str, type]] = None
    ) -> Generator:
    """Generates batches of an array stored across multiple files.  

    Args:
        dir (str): Directory from which to read files.
        prefix (str): File name prefix. 
        batch_size (Optional[int], optional): Optionally load array in batches 
            a certain size. Defaults to None.
        sort_by (Optional[Tuple[str, type]], optional): Sort by a field `sort_by[0]`
            having type `sort_by[1]`. Defaults to None.

    Yields:
        Generator: A generator that yields rows of the array. 
    """

    # Get sorted list of files
    files = [f for f in os.listdir(dir) if f.startswith(prefix) and f.endswith('.npy')]
    if sort_by is not None: 
        fields = [sort_by[1](get_field_from_fname(f, sort_by[0])) for f in files]
        pairs = sorted(zip(files, fields), key=lambda x: x[1])
        files, _ = zip(*pairs)
        files = list(files)
    else: 
        files = sorted(files)

    if batch_size is None:  # each batch is a file
        for f in files:
            yield np.load(os.path.join(dir, f))

    else: # each batch is of desired size
        leftover = None
        for f in files: 
            array = np.load(os.path.join(dir, f))
            leftover = array if leftover is None else np.concatenate((leftover, array))
            while len(leftover) >= batch_size:
                yield leftover[:batch_size]
                leftover = leftover[batch_size:]
        if leftover is not None and len(leftover) > 0: 
            yield leftover



def gen_tensors_as_arrays(
        dir: str, 
        prefix: str, 
        batch_size: Optional[int] = None, 
        sort_by: Optional[Tuple[str, type]] = None
    ) -> Generator:
    loader = gen_tensors(dir, prefix, batch_size, sort_by)
    for tensor in loader: 
        yield tensor.numpy()


def get_generator(gen_fcn: Callable, *args, **kwargs) -> Generator:
    return gen_fcn(*args, **kwargs)


def read_tensors(
        dir: str, 
        prefix: str, 
        batch_size: Optional[int] = None, 
        sort_by: Optional[Tuple[str, type]] = None
    ) -> torch.Tensor:
    tensor_list = []
    tensor_loader = gen_tensors(dir, prefix, batch_size, sort_by)
    for tensor in tensor_loader: 
        tensor_list.append(tensor)
    return torch.cat(tensor_list)


def refresh_directory(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


def remove_directory(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)


def remove_file(path):
    if os.path.exists(path):
        os.remove(path)


def write_rows_to_csv(path: str, rows: List[Dict]) -> None:

    if not rows:
        raise ValueError("The list of rows is empty.")

    header = rows[0].keys()
    file_exists = os.path.isfile(path)

    with open(path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def load_yaml(path: str) -> Dict:
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def write_yaml(data: Dict, path: str) -> None:
    with open(path, 'w') as file:
        yaml.dump(data, file)


def multiply_list(list_: List[Union[int, float]]):
    out = 1
    for el in list_:
        out *= el
    return out


def init_process(
        rank: int, 
        world_size: int, 
        fcn: Callable, 
        path_shared: str,
        backend: str,
        **kwargs
    ):
    dist.init_process_group(
        backend, init_method=f'file://{path_shared}',
        rank=rank, world_size=world_size
    )
    fcn(rank, world_size, **kwargs)


def execute_script(path: str, flags: Dict[str, str], raise_error: bool = True) -> int:

    # Compile arguments for subprocess.run()
    args = [sys.executable, path]
    for k, v in flags.items():
        args.append(f'--{k}')
        if v is not None:
            if isinstance(v, list):
                for item in v:
                    args.append(str(item))
            else:
                args.append(str(v))

    # Run script, (optionally) raising an error if encountered
    try:
        proc = subprocess.run(args, capture_output=True, check=True, text=True)
        print(proc.stdout)
        return 0
    except subprocess.CalledProcessError as err: 
        print(f"returncode = {err.returncode} \n"
              f"stderr = {err.stderr}")
        if raise_error:
            raise Exception(err)
        else: 
            return err.returncode


def l2_norm(input: Union[torch.Tensor, np.ndarray]) -> float:
    if isinstance(input, torch.Tensor):
        return (torch.sqrt(torch.sum(input ** 2) / torch.numel(input))).item()
    elif isinstance(input, np.ndarray):
        return np.sqrt(np.sum(input ** 2) / input.size)
    else: 
        raise TypeError("Input must be a torch.Tensor or a np.ndarray")


def is_square(n):
    if n < 0:
        return False
    sqrt_n = int(math.sqrt(n))
    return sqrt_n * sqrt_n == n


def safe_l2_normalization(
        input: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
    norm = l2_norm(input)
    if norm > 0:
        input = input / norm
    return input


def model_from_loads(loads):
    model = LowRankCovariance(loads.shape[0], loads.shape[1])
    model.set_loads(loads)
    return model


def get_indices_from_grid_shape(grid_shape: torch.Tensor):
    indices = []
    for sz in grid_shape:
        ind = torch.arange(sz, dtype=torch.int32)
        indices.append(ind)
    return torch.cartesian_prod(*indices)


def get_points_from_grid_shape(grid_shape: torch.Tensor):
    points = []
    for sz in grid_shape:
        p = torch.arange(sz, dtype=torch.int32) / sz
        points.append(p)
    return torch.cartesian_prod(*points)


class ReshapingIndexMap(object):
    """Given an `old_shape` and a `new_shape`, the ReshapingIndexMap may
    be used to map old indices to new indices when a tensor is reshaped like
        >>> tensor = torch.zeros(old_shape)
        >>> tensor = tensor.reshape(new_shape)

    The class' map and seq_maps method is based on the fact that 
    `torch.reshape` behaves as below:
        >>> old_shape = [2, 3, 2, 3]
        >>> new_shape = [6, 6]
        >>> tensor = torch.arange(36).reshape(old_shape)
        >>> tensor
        tensor([[[[ 0,  1,  2],
                [ 3,  4,  5]],

                [[ 6,  7,  8],
                [ 9, 10, 11]],

                [[12, 13, 14],
                [15, 16, 17]]],


                [[[18, 19, 20],
                [21, 22, 23]],

                [[24, 25, 26],
                [27, 28, 29]],

                [[30, 31, 32],
                [33, 34, 35]]]])
        >>> tensor.reshape(new_shape)
        tensor([[ 0,  1,  2,  3,  4,  5],
                [ 6,  7,  8,  9, 10, 11],
                [12, 13, 14, 15, 16, 17],
                [18, 19, 20, 21, 22, 23],
                [24, 25, 26, 27, 28, 29],
                [30, 31, 32, 33, 34, 35]])
    """

    def __init__(self, old_shape: List[int], new_shape: List[int]) -> None:
        self.ndim_old = len(old_shape)
        self.ndim_new = len(new_shape)
        self.cum_prods = [None] * self.ndim_old
        cum_prod = 1
        i = self.ndim_old - 1
        while i >= 0:
            self.cum_prods[i] = cum_prod
            cum_prod *= old_shape[i]
            i -= 1
        self.new_shape = new_shape
    
    def map(
            self, 
            old_idx: Union[torch.Tensor, List[int]]
        ) -> Union[torch.Tensor, List[int]]:

        is_tensor = torch.is_tensor(old_idx)
        if is_tensor:
            old_idx = old_idx.tolist()

        quotient = 0
        for i in range(self.ndim_old):
            quotient += old_idx[i] * self.cum_prods[i]

        new_idx = [None] * self.ndim_new
        remainder = None
        i = self.ndim_new - 1
        while i >= 0:
            quotient, remainder = divmod(quotient, self.new_shape[i])
            new_idx[i] = remainder
            i -= 1

        return torch.tensor(new_idx, dtype=torch.int32) if is_tensor else new_idx
    
    def seq_map(
            self,
            old_idx: Union[torch.Tensor, List[List[int]]]
        ) -> Union[torch.Tensor, List[List[int]]]:

        is_tensor = torch.is_tensor(old_idx)
        seq_len = len(old_idx)

        out = [None] * seq_len
        for i in range(seq_len):
            out[i] = self.map(old_idx[i])

        if is_tensor:
            return torch.row_stack(out)
        else:
            return out
        
        
class FlatMaskIndexMap(object):

    def __init__(self, mask: torch.Tensor) -> None:
        mask = torch.flatten(mask)
        idx_unmask = torch.arange(multiply_list(mask.shape))[mask]
        idx_mask = torch.arange(len(idx_unmask))
        self.idx_map = pd.DataFrame(
            torch.stack((idx_unmask, idx_mask), dim=0).t().numpy(),
            columns=['key', 'value']
        )

    def __call__(self, idx_unmask: torch.Tensor) -> torch.Tensor:
        idx_unmask = pd.DataFrame(idx_unmask.numpy(), columns=['key'])
        idx_mask = pd.merge(idx_unmask, self.idx_map, on='key', how='left')
        return torch.tensor(idx_mask['value'].values)
        

def procrustes_rotation(input, target):
    """Supposing `input` and `target` are both K-by-T matrices, finds the 
    K-by-K rotation matrix `rot_mat` such that `rot_mat @ input` is close to 
    `target`. Supports both torch.Tensor and numpy.ndarray as inputs."""
    
    # Check if the input is a NumPy array or a PyTorch tensor
    if isinstance(input, np.ndarray) and isinstance(target, np.ndarray):
        cov = target @ input.T
        U, _, Vt = np.linalg.svd(cov)
        rot_mat = U @ Vt
        
    elif isinstance(input, torch.Tensor) and isinstance(target, torch.Tensor):
        cov = target @ input.T
        U, _, Vt = torch.linalg.svd(cov)
        rot_mat = U @ Vt
    
    else:
        raise TypeError("Both input and target must be of the same type: either both NumPy arrays or both PyTorch tensors.")
    
    return rot_mat


# -------------------- OBJECTIVE FUNCTIONS -------------------- #

def loss_fcn(preds, cov):
    return torch.sum((preds - cov) ** 2) / len(preds)


def penalty_fcn(loads: torch.Tensor, alpha: float, diff_mat: torch.Tensor):
    n_vars, n_facs = loads.shape
    return alpha * torch.trace(loads.t() @ diff_mat @ loads) / n_vars / n_facs


def penalty_fcn_gradient(loads: torch.Tensor, diff_mat: torch.Tensor):
    n_vars, n_facs = loads.shape
    return 2 * diff_mat @ loads / n_vars / n_facs


def compute_loss(model, dir_cov, split, fold=None):
    points_loader = gen_tensors(dir_cov, 'points_', sort_by=('i', int))
    cov_prefix = f'cov_split-{split}_'
    if fold is not None: 
        cov_prefix += f'v-{fold}_i-'
    else: 
        cov_prefix += f'i-'
    cov_loader = gen_tensors(dir_cov, cov_prefix, sort_by=('i', int))
    loss = 0
    with torch.no_grad():
        for points, cov in zip(points_loader, cov_loader):
            preds = model(points)
            loss += loss_fcn(preds, cov)
    return loss



# -------------------- SPARSE MATRICES -------------------- #

def reshape_sparse_coo_tensor(
        tensor: torch.Tensor,
        new_sz: Sequence[int]
    ) -> torch.Tensor:
    tensor = tensor.coalesce()
    new_idx = ReshapingIndexMap(
        old_shape=tensor.size(),
        new_shape=new_sz
    ).seq_map(tensor.indices().t()).t()
    return torch.sparse_coo_tensor(
        indices=new_idx, 
        values=tensor.values(),
        size=new_sz
    )

        
def slice_sparse_coo_tensor(
        tensor: torch.Tensor, 
        dim: int, 
        start: int, 
        stop: int
    ) -> torch.Tensor:

    # Extract relevant information from full tensor
    tensor = tensor.coalesce()
    indices = tensor.indices()
    values = tensor.values()
    new_sz = list(tensor.size())

    # Filter indices for desired slice range
    mask = (indices[dim] >= start) & (indices[dim] < stop)
    new_indices = indices[:, mask]
    new_values = values[mask]

    # Adjust indices and shape for sliced range
    new_indices[dim] -= start
    new_sz[dim] = stop - start

    return torch.sparse_coo_tensor(
        indices=new_indices,
        values=new_values,
        size=new_sz
    )


OFF_DIAG_SHIFTS = {

    1: [
        torch.tensor([-1], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32)
    ],

    2: [
        torch.tensor([0, -1], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),

        torch.tensor([1, 0], dtype=torch.int32),
        torch.tensor([1, -1], dtype=torch.int32),
        torch.tensor([1, 1], dtype=torch.int32),


        torch.tensor([-1, 0], dtype=torch.int32),
        torch.tensor([-1, -1], dtype=torch.int32),
        torch.tensor([-1, 1], dtype=torch.int32),
    ],

    3: [
        torch.tensor([0, 0, -1], dtype=torch.int32),
        torch.tensor([0, 0, 1], dtype=torch.int32),
        torch.tensor([0, -1, 0], dtype=torch.int32),
        torch.tensor([0, -1, -1], dtype=torch.int32),
        torch.tensor([0, -1, 1], dtype=torch.int32),
        torch.tensor([0, 1, 0], dtype=torch.int32),
        torch.tensor([0, 1, -1], dtype=torch.int32),
        torch.tensor([0, 1, 1], dtype=torch.int32),

        torch.tensor([1, 0, 0], dtype=torch.int32),
        torch.tensor([1, 0, -1], dtype=torch.int32),
        torch.tensor([1, 0, 1], dtype=torch.int32),
        torch.tensor([1, -1, 0], dtype=torch.int32),
        torch.tensor([1, -1, -1], dtype=torch.int32),
        torch.tensor([1, -1, 1], dtype=torch.int32),
        torch.tensor([1, 1, 0], dtype=torch.int32),
        torch.tensor([1, 1, -1], dtype=torch.int32),
        torch.tensor([1, 1, 1], dtype=torch.int32),

        torch.tensor([-1, 0, 0], dtype=torch.int32),
        torch.tensor([-1, 0, -1], dtype=torch.int32),
        torch.tensor([-1, 0, 1], dtype=torch.int32),
        torch.tensor([-1, -1, 0], dtype=torch.int32),
        torch.tensor([-1, -1, -1], dtype=torch.int32),
        torch.tensor([-1, -1, 1], dtype=torch.int32),
        torch.tensor([-1, 1, 0], dtype=torch.int32),
        torch.tensor([-1, 1, -1], dtype=torch.int32),
        torch.tensor([-1, 1, 1], dtype=torch.int32),
    ],
}


def create_second_difference_matrix(grid_shape):
    """Creates a square matricized second difference sparse tensor for 
    `grid_shape`."""

    # Over-allocate memory for `idx` and `vals`.
    # Note that each interior diag cell touches 3^d - 1 off-diag cells.
    ndim = len(grid_shape)
    n_vars = multiply_list(grid_shape)
    fill_val = -2
    idx = torch.full((2, n_vars * 3 ** ndim), fill_val, dtype=torch.int32)
    vals = torch.full((n_vars * 3 ** ndim,), fill_val, dtype=torch.float32)

    # Build `idx` and `vals`
    idx_map = ReshapingIndexMap(grid_shape + grid_shape, [n_vars, n_vars])
    idx_grid = get_indices_from_grid_shape(grid_shape)
    diag_val = 3 ** ndim - 1
    off_diag_shifts = OFF_DIAG_SHIFTS[ndim]
    cnt = 0
    for i in range(len(idx_grid)):

        # Get `base_idx`
        base_idx = idx_grid[i]
        if ndim == 1:
            base_idx = base_idx.reshape(1)        

        # Add diagonal
        diag_idx = torch.cat((base_idx, base_idx))
        idx[:,cnt] = idx_map.map(diag_idx)
        vals[cnt] = diag_val
        cnt += 1

        # Add off-diagonals
        for shift in off_diag_shifts:

            shift_idx = base_idx - shift
            interior = torch.all(
                (shift_idx >= torch.zeros(ndim)) &
                (shift_idx < torch.tensor(grid_shape))
            )
            if interior:
                off_diag_idx = torch.cat((base_idx, shift_idx))
                idx[:,cnt] = idx_map.map(off_diag_idx)
                vals[cnt] = -1
                cnt += 1

    # Create sparse matrix
    diff_mat = torch.sparse_coo_tensor(
        indices=idx[:,:cnt], 
        values=vals[:cnt],
        size=[n_vars, n_vars]
    )
                
    return diff_mat



# -------------------- DATA PREP -------------------- #

def flatten_dataset(
        dir_dataset: str, 
        dir_out: str, 
        sub_num: int,
        bsz_time: int, 
        bsz_space: int,
        path_mask: Optional[str]
    ) -> Tuple[int, List[int]]: 
    """Creates two flat views (batched by row) of the dataset represented by
    tensors of shape (T, M_1, ..., M_D) in `dir_dataset`: 
        (1) A dataset with shape (T, M_1*...*M_D)
        (2) A dataset with shape (M_1*...*M_D, T)
    Finally, returns `n_time` and `sz_space`.
    """

    # Optionally load mask
    mask = None if path_mask is None else torch.flatten(torch.load(path_mask))

    # Create the first flat view (temporal rows)
    i = 0
    n_time = 0
    for data in gen_tensors(dir_dataset, f'data_n-{sub_num}_', bsz_time, ('i', int)):

        if i == 0: 
            sz_space = list(data.shape[1:])
            if mask is None: 
                n_space = multiply_list(sz_space)
            else: 
                n_space = torch.sum(mask).item()

        sz = len(data)
        data = data.reshape(sz, multiply_list(sz_space))

        if mask is not None: 
            data = data[:,mask]

        path = os.path.join(dir_out, f'data-time_split-full_n-{sub_num}_i-{i}_.pt')
        torch.save(data, path)

        i += 1
        n_time += sz

    # Create the second flat view (spatial rows)
    i = 0
    start = 0
    while start < n_space:

        sz = min(bsz_space, n_space - start)
        batches = []

        for batch in gen_tensors(
            dir_out, f'data-time_split-full_n-{sub_num}_', 
            bsz_time, ('i', int)
            ):
            batches.append(batch[:, start:(start + sz)])
        data = torch.cat(batches).t()

        path = os.path.join(dir_out, f'data-space_split-full_n-{sub_num}_i-{i}_.pt')
        torch.save(data, path)

        i += 1
        start += sz

    return n_time, sz_space



# -------------------- TRAINING POINTS -------------------- #

def gen_cartesian_prod(input: torch.Tensor) -> Generator:
    """Yields elements of the cartesian product input x input in batches of
    size len(input)."""
    grid_size = len(input)
    for i in range(0, grid_size):
        idx = torch.cartesian_prod(
            torch.tensor([i], dtype=torch.int32),
            torch.arange(grid_size, dtype=torch.int32)
        )
        yield torch.column_stack((input[idx[:,0]], input[idx[:,1]]))


def gen_points(
        grid_shape: List[int],
        delta: float, 
        batch_size: int,
        off_band: bool = True,
        exclude_upp_tri: bool = True,
        as_numpy: bool = False,
        path_mask: Optional[str] = None
    ) -> Generator:
    """Yields square matricized training points for a grid_shape-by-grid_shape 
    covariance tensor in batches."""

    ndim = len(grid_shape)

    # Batch size check
    if path_mask is None: 
        min_bsz = multiply_list(grid_shape)
    else: 
        min_bsz = torch.nonzero(torch.load(path_mask)).shape[0]
    if batch_size < min_bsz: 
        raise Exception(f"Must have batch_size >= {min_bsz}")

    # Get (optionally masked) spatial indices and bandwidths
    indices = get_indices_from_grid_shape(grid_shape)
    if path_mask is not None:

        # Get map from flattened unmasked indices to flattened masked indices
        mask = torch.load(path_mask)
        idx_map_mask = FlatMaskIndexMap(mask)
        
        # Mask indices
        nz_set = {tuple(row.tolist()) for row in torch.nonzero(mask)}
        indices = torch.stack([row for row in indices if tuple(row.tolist()) in nz_set])
    
    bandwidths = torch.tensor([math.ceil(grid_shape[d]*delta) for d in range(ndim)])

    idx_map = ReshapingIndexMap(
        old_shape=grid_shape + grid_shape, 
        new_shape=[multiply_list(grid_shape), multiply_list(grid_shape)]
    )
    start_new_batch = True
    leftovers = None
    for cp_batch in gen_cartesian_prod(indices): 

        if start_new_batch:
            start_new_batch = False
            batch = torch.zeros(batch_size, 2, dtype=torch.int32)
            start_idx = 0
            num_leftovers = len(leftovers) if leftovers is not None else 0
            if num_leftovers > 0:
                batch[:num_leftovers] = leftovers
                start_idx = num_leftovers
                leftovers = None

        # Keep only the points on/off the band
        dists = torch.abs(cp_batch[:,ndim:(2*ndim)] - cp_batch[:,0:ndim])
        if off_band:
            keep = torch.any(dists > bandwidths, dim=1)
        else: 
            keep = torch.all(dists <= bandwidths, dim=1)
            
        # Keep only the points in the lower triangle
        if torch.sum(keep) > 0:
            cp_batch = idx_map.seq_map(cp_batch[keep])  # square matricize indices
            if exclude_upp_tri:
                keep = cp_batch[:,0] >= cp_batch[:,1]
                cp_batch = cp_batch[keep]
        else:
            continue

        # Continue if all points filtered from batch
        num_to_keep = cp_batch.size(0)
        if num_to_keep == 0:
            continue

        # Convert to masked indices
        if path_mask is not None: 
            cp_batch[:,0] = idx_map_mask(cp_batch[:,0])
            cp_batch[:,1] = idx_map_mask(cp_batch[:,1])

        num_to_inc = min(num_to_keep, batch_size - start_idx)
        num_to_exc = max(0, num_to_keep - num_to_inc)
        batch[start_idx:(start_idx + num_to_inc)] = cp_batch[:num_to_inc]
        start_idx += num_to_inc
        
        # Iteration Cases: 
        #  [Any]
        #   (1) cb_batch overfills batch --> start_idx == batch_size and num_to_exc > 0
        #   (2) cb_batch precisely fills batch --> start_idx == batch_size and num_to_exc < 0
        #  [Last]
        #   (3) cb_batch underfills batch --> start_idx < batch_size
        if start_idx == batch_size:  # if (1) or (2), yield saturated batch
            yield batch.numpy() if as_numpy else batch
            start_new_batch = True
            if num_to_exc > 0:
                leftovers = cp_batch[-num_to_exc:]
    
    if leftovers is not None:  # if (2), yield leftovers
        yield leftovers.numpy() if as_numpy else leftovers
    elif start_idx < batch_size:  # if (3), yield underfilled batch
        yield batch[:start_idx].numpy() if as_numpy else batch[:start_idx]


# TODO: Replace this with calls to Redis or wrap in njit...
def compute_covariance(
        points: Union[torch.Tensor, np.ndarray], 
        dir_data: str, 
        split: str
    ) -> Union[torch.Tensor, np.ndarray]:

    # If points is a numpy array, we will return a numpy array
    as_numpy = False
    if isinstance(points, np.ndarray):
        as_numpy = True
        points = torch.tensor(points)

    n_points = len(points)
    t1 = torch.zeros(n_points, dtype=torch.float32)
    t2 = torch.zeros(n_points, dtype=torch.float32)
    t3 = torch.zeros(n_points, dtype=torch.float32)
    n = 0
    data_loader = gen_tensors(dir_data, f'data-time-{split}')
    for data in data_loader: 

        n += len(data)
        for i in range(n_points): 
            row, col = points[i]
            t1[i] += torch.sum(data[:,row] * data[:,col])
            t2[i] += torch.sum(data[:,row])
            t3[i] += torch.sum(data[:,col])

    cov = (t1 - t2 * t3 / n) / (n - 1)
    return cov.numpy() if as_numpy else cov

