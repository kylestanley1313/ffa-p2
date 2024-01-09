import argparse
import os
import math
import shutil
import torch
from abc import ABC, abstractmethod
from typing import List

from utils import write_generated_tensor


# ---------- UTILITIES ---------- #

def op_facs_loads(loads: torch.Tensor, facs: torch.Tensor, ndim: int) -> torch.Tensor:
    """Performs the operation `sum_k(facs[k]*loads[k])."""
    if ndim == 1:
        raise NotImplementedError
    elif ndim == 2:
        raise NotImplementedError
    elif ndim == 3:
        return torch.einsum('nk,kabc->nabc', facs, loads)
    else: 
        raise NotImplementedError 
    

class BumpFunction1D(object):

    def __init__(
            self, 
            center: float,
            scale: float,
            max: float,
        ):
        self.ndim = 1
        self.center = center
        self.scale = scale
        self.max = max

    def __call__(self, points: torch.Tensor):
        
        # Center then scale points about origin
        points = points.clone()
        points -= self.center
        points /= self.scale

        # Evaluate transformed points
        vals = torch.zeros_like(points, dtype=torch.float64)
        idx = torch.abs(points) <= 1
        vals[idx] = torch.exp(-1 / (1 - points[idx] ** 2))
        return self.max * math.exp(1) * vals


class BumpFunction2D(object):

    def __init__(
            self, 
            center: List[float],
            rotation: float,
            scale: List[float],
            max: float
        ):
        self.ndim = 2
        self.center = center
        self.max = max

        # Matrix for rotation about origin
        rotation = rotation * math.pi / 180  # degrees to radians
        self.rot_mat = torch.tensor([
            [math.cos(rotation), -math.sin(rotation)],
            [math.sin(rotation), math.cos(rotation)]
        ], dtype=torch.float64)

        # Matrix for scaling about origin
        self.scale_mat = torch.tensor([
            [scale[0], 0],
            [0, scale[1]]
        ], dtype=torch.float64)

    def __call__(self, points: torch.Tensor):

        # Center, then rotate and scale about origin
        points -= torch.tensor(self.center)
        points = (torch.inverse(self.rot_mat @ self.scale_mat) @ points.t()).t()

        # Evaluate transformed points
        vals = torch.zeros(len(points), dtype=torch.float64)
        r = torch.sqrt(torch.sum(points ** 2, dim=1))
        idx = torch.abs(r) <= 1
        vals[idx] = torch.exp(-1 / (1 - r[idx] ** 2))
        return self.max * math.exp(1) * vals
    

class BumpFunction3D(object):

    def __init__(
            self, 
            center: List[float],
            rotation: List[float],
            scale: List[float],
            max: float
        ):
        self.ndim = 3
        self.center = center
        self.max = max

        # Matrix for rotation about origin
        rotation = [r * math.pi / 180 for r in rotation]  # degrees to radians
        rot_mat_x = torch.tensor([
            [1, 0, 0],
            [0, math.cos(rotation[0]), -math.sin(rotation[0])],
            [0, math.sin(rotation[0]), math.cos(rotation[0])]
        ], dtype=torch.float64)
        rot_mat_y = torch.tensor([
            [math.cos(rotation[1]), 0, math.sin(rotation[1])],
            [0, 1, 0],
            [-math.sin(rotation[1]), 0, math.cos(rotation[1])]
        ], dtype=torch.float64)
        rot_mat_z = torch.tensor([
            [math.cos(rotation[2]), -math.sin(rotation[2]), 0],
            [math.sin(rotation[2]), math.cos(rotation[2]), 0],
            [0, 0, 1]
        ], dtype=torch.float64)
        self.rot_mat = rot_mat_z @ rot_mat_y @ rot_mat_x

        # Matrix for scaling about origin
        self.scale_mat = torch.tensor([
            [scale[0], 0, 0],
            [0, scale[1], 0],
            [0, 0, scale[2]]
        ], dtype=torch.float64)

    def __call__(self, points: torch.Tensor) -> torch.Tensor:

        # Center, then rotate and scale about origin
        points = points.clone()
        points -= torch.tensor(self.center)
        points = (torch.inverse(self.rot_mat @ self.scale_mat) @ points.t()).t()

        # Evaluate transformed points
        vals = torch.zeros(len(points), dtype=torch.float64)
        r = torch.sqrt(torch.sum(points ** 2, dim=1))
        idx = torch.abs(r) <= 1
        vals[idx] = torch.exp(-1 / (1 - r[idx] ** 2))
        return self.max * math.exp(1) * vals



# ---------- LOADING FUNCTIONS ---------- #
# NOTE: All loading functions are defined on [0,1]^3
    
class LoadingFunction(ABC):
    """Base class for LoadingFunctions which build a loading function of 
    dimension `ndim` from `pieces`. When called, a LoadingFunction realizes
    itself on some grid defined by `points`."""

    def __init__(self):
        if not self._compatible_pieces():
            raise Exception("Pieces do not having matching dimensions!")

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        vals = torch.zeros(len(points), dtype=torch.float64)
        for piece in self.pieces:
            vals += piece(points)
        return vals
    
    @property
    @abstractmethod
    def pieces(self) -> List:
        pass

    @property
    @abstractmethod
    def ndim(self) -> int:
        pass

    def _compatible_pieces(self):
        ndims = []
        for piece in self.pieces:
            ndims.append(piece.ndim)
        return all(d == self.ndim for d in ndims)


class CornerPair3D1(LoadingFunction):

    ndim = 3
    pieces = [
        BumpFunction3D([0.25, 0.25, 0.25], [0, 0, 0], [0.15, 0.15, 0.15], 1),
        BumpFunction3D([0.75, 0.75, 0.75], [0, 0, 0], [0.15, 0.15, 0.15], 1),
    ]


class CornerPair3D2(LoadingFunction):

    ndim = 3
    pieces = [
        BumpFunction3D([0.25, 0.25, 0.75], [0, 0, 0], [0.15, 0.15, 0.15], 1),
        BumpFunction3D([0.75, 0.75, 0.25], [0, 0, 0], [0.15, 0.15, 0.15], 1),
    ]


class CornerPair3D3(LoadingFunction):

    ndim = 3
    pieces = [
        BumpFunction3D([0.25, 0.75, 0.25], [0, 0, 0], [0.15, 0.15, 0.15], 1),
        BumpFunction3D([0.75, 0.25, 0.75], [0, 0, 0], [0.15, 0.15, 0.15], 1),
    ]


class CornerPair3D4(LoadingFunction):

    ndim = 3
    pieces = [
        BumpFunction3D([0.75, 0.25, 0.25], [0, 0, 0], [0.15, 0.15, 0.15], 1),
        BumpFunction3D([0.25, 0.25, 0.75], [0, 0, 0], [0.15, 0.15, 0.15], 1),
    ]



# ---------- LOADING SCHEMES ---------- #
    
class LoadingScheme(ABC):
    """Base clas for LoadingSchemes which use `loading_fcns` and `scales` to 
    build a set of scaled loading functions. When called, a LoadingScheme 
    realizes itself on a grid defined by `points`."""

    def __init__(self):
        if not self._compatible_fcns():
            raise Exception("Loading functions do not having matching dimensions!")
        if not len(self.loading_fcns) == len(self.scales):
            raise Exception("Number of loading functions differs from number of scales!")
        
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        vals = torch.zeros(self.ncomps, len(points), dtype=torch.float64)
        for k in range(self.ncomps):
            fcn = self.loading_fcns[k]()
            vals[k] = fcn(points) * self.scales[k]
        return vals

    @property
    @abstractmethod
    def loading_fcns(self):
        pass

    @property
    @abstractmethod
    def scales(self):
        pass

    @property
    @abstractmethod
    def ndim(self):
        pass

    @property
    def ncomps(self):
        return len(self.loading_fcns)

    def _compatible_fcns(self):
        ndims = []
        for fcn in self.loading_fcns:
            ndims.append(fcn.ndim)
        return all(d == self.ndim for d in ndims)
        

class BumpScheme1(LoadingScheme):

    ndim = 3
    loading_fcns = [
        CornerPair3D1, 
        CornerPair3D2,
        CornerPair3D3,
        CornerPair3D4
    ]
    scales = [4, 3, 2, 1]


LOADING_SCHEMES = {
    'BumpScheme1': BumpScheme1
}


# ---------- MAIN FUNCTIONS ---------- #

def build_loadings(
        indices: torch.Tensor, 
        grid_shape: List[int], 
        vals: torch.Tensor
    ) -> torch.Tensor:

    ndim = len(grid_shape)
    ncomps = vals.shape[0]
    loads = torch.zeros(ncomps, *grid_shape, dtype=torch.float64)

    if ndim == 1:
        loads[:,indices] = vals
    elif ndim == 2:
        loads[:, indices[:,0], indices[:,1]] = vals
    elif ndim == 3:
        loads[:, indices[:,0], indices[:,1], indices[:,2]] = vals
    else: 
        raise Exception("Length of `grid_size` must be less than or equal to 3!")

    return loads


def simulate_ffm_data(
        loads: torch.Tensor,
        err_sd: int, 
        num_train: int,
        num_val: int,
        batch_size: int,
        gen: torch.Generator = torch.Generator(),
    ):
    num_facs = loads.shape[0]
    grid_shape = loads.shape[1:]
    n = num_train + num_val
    while n > 0:
        n_batch = min(batch_size, n) 
        facs = torch.normal(0, 1, (n_batch, num_facs), dtype=torch.float64, generator=gen)
        errs = torch.normal(0, 1, (n_batch, *grid_shape), dtype=torch.float64, generator=gen)
        errs *= err_sd
        data = op_facs_loads(loads, facs, len(grid_shape)) + errs
        yield data
        n -= n_batch


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir',
        help="Directory in which simulated data will be stored."
    )
    parser.add_argument(
        '--grid_shape', nargs='+', type=int,
        help="Shape of the grid on which to simulate data."
    )
    parser.add_argument(
        '--load_scheme', type=str,
        help="Loading scheme used to simulate data."
    )
    parser.add_argument(
        '--num_train', type=int,
        help="Number of training samples to simulate."
    )
    parser.add_argument(
        '--num_val', type=int,
        help="Number of validation samples to simulate."
    )
    parser.add_argument(
        '--batch_size', type=int,
        help="Maximum number of samples per output file."
    )
    parser.add_argument(
        '--seed', default=12345,
        help="Integer used to seed generator."
    )
    args = parser.parse_args()

    # Configure globals
    load_scheme = LOADING_SCHEMES[args.load_scheme]()
    gen = torch.Generator().manual_seed(args.seed)

    # Check for `load_scheme` and `grid_shape` compatibility
    if load_scheme.ndim != len(args.grid_shape):
        msg = ("Number of loading scheme dimensions does not match the number " 
               "of grid dimensions!")
        raise Exception(msg)       


    # ---------- LOADING PREP ---------- #

    print("Preparing loadings...")

    # Delete files from directory
    out_dir = os.path.join('.', 'datasets', args.dir)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # Generate `points` and `indices` from `grid_shape`
    points = []
    indices = []
    for sz in args.grid_shape:
        indices_ = torch.arange(sz, dtype=torch.int32)
        points_ = indices_.to(torch.float64) / sz
        indices.append(indices_)
        points.append(points_)
    indices = torch.cartesian_prod(*indices)  # n-by-ndim
    points = torch.cartesian_prod(*points)    # n-by-ndim

    # Build loading tensor
    vals = load_scheme(points)  # ncomps-by-n
    loads = build_loadings(indices, args.grid_shape, vals)

    # ---------- ERROR PREP ---------- #

    # TODO: Error preparation


    # ---------- DATA SIMULATION ---------- #

    print("Simulating new data...")

    err_sd = 0.2
    data = simulate_ffm_data(
        loads, 
        err_sd,
        num_train=args.num_train,
        num_val=args.num_val,
        batch_size=args.batch_size,
        gen=gen
    )

    write_generated_tensor(data, out_dir, 'data')

    print("DONE!")
