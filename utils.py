import csv
import math
import os
import shutil
import torch
import yaml
from typing import Dict, Generator, List, Union



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
    

def write_generated_tensor(tensor_loader: Generator, dir: str, prefix: str):
    for i, batch in enumerate(tensor_loader):
        path = os.path.join(dir, f'{prefix}-{i}.pt')
        torch.save(batch, path)


def read_tensors(dir: str, prefix: str) -> Generator:
    files = sorted(os.listdir(dir))
    for f in files:
        if f.startswith(prefix) and f.endswith('.pt'):
            yield torch.load(os.path.join(dir, f))


def refresh_directory(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


def remove_file(path):
    if os.path.exists(path):
        os.remove(path)


def write_rows_to_csv(path, rows):
    mode = 'a' if os.path.exists(path) else 'w'
    with open(path, mode, newline='') as file:
        writer = csv.writer(file)
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
        


# -------------------- DIFFERENCE MATRICES -------------------- #


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
    num_vars = multiply_list(grid_shape)
    fill_val = -2
    idx = torch.full((2, num_vars * 3 ** ndim), fill_val, dtype=torch.int32)
    vals = torch.full((num_vars * 3 ** ndim,), fill_val, dtype=torch.float64)

    # Build `idx` and `vals`
    idx_map = ReshapingIndexMap(grid_shape + grid_shape, [num_vars, num_vars])
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
        size=[num_vars, num_vars]
    )
                
    return diff_mat



# -------------------- FLATTENING -------------------- #

def flatten_dataset(dir_in: str, dir_out: str) -> List[int]:
    """Reads a potentially unflattened dataset from `dir_in`, flattens it, 
    then writes the result to `dir_out`."""

    files = [f for f in os.listdir(dir_in) if f.endswith('.pt')]
    for i in range(len(files)):
        path_in = os.path.join(dir_in, files[i])
        data = torch.load(path_in)
        
        # Get `grid_shape` and `num_vars` from first data file
        if i == 0:
            grid_shape = list(data.shape[1:])
            num_vars = multiply_list(grid_shape)
        
        data = data.reshape(len(data), num_vars)
        path_out = os.path.join(dir_out, files[i])
        torch.save(data, path_out)

    return grid_shape



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


def gen_points(grid_shape: List[int], delta: float, batch_size: int) -> Generator:
    """Yields square matricized training points for a grid_shape-by-grid_shape 
    covariance tensor in batches."""

    ndim = len(grid_shape)
    num_vars = multiply_list(grid_shape)

    if batch_size < num_vars: 
        raise Exception("Must have batch_size >= num_vars")

    indices = get_indices_from_grid_shape(grid_shape)
    bandwidths = torch.tensor([math.ceil(grid_shape[d]*delta) for d in range(ndim)])

    idx_map = ReshapingIndexMap(grid_shape + grid_shape, [num_vars, num_vars])
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

        keep = cp_batch[:,ndim:(2*ndim)] < cp_batch[:,0:ndim] - bandwidths
        keep = torch.all(keep, dim=1)

        num_to_keep = keep.sum().item()
        if num_to_keep == 0:
            continue

        num_to_inc = min(num_to_keep, batch_size - start_idx)
        num_to_exc = max(0, num_to_keep - num_to_inc)
        cp_batch = idx_map.seq_map(cp_batch[keep])
        batch[start_idx:(start_idx + num_to_inc)] = cp_batch[:num_to_inc]
        start_idx += num_to_inc
        
        # Iteration Cases: 
        #  [Any]
        #   (1) cb_batch overfills batch --> start_idx == batch_size and num_to_exc > 0
        #   (2) cb_batch precisely fills batch --> start_idx == batch_size and num_to_exc < 0
        #  [Last]
        #   (3) cb_batch underfills batch --> start_idx < batch_size
        if start_idx == batch_size:  # if (1) or (2), yield saturated batch
            yield batch
            start_new_batch = True
            if num_to_exc > 0:
                leftovers = cp_batch[-num_to_exc:]
    
    if leftovers is not None:  # if (2), yield leftovers
        yield leftovers
    elif start_idx < batch_size:  # if (3), yield underfilled batch
        yield batch[:start_idx]
