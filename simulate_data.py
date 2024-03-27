import argparse
import os
import math
import shutil
import torch
from abc import ABC, abstractmethod
from typing import List, Union

from config import load_config
from utils.utils import refresh_directory, safe_normalize, write_generated_tensor


# ---------- UTILITIES ---------- #

def nsamp_basis_comb(basis: Union[torch.Tensor], coeffs: torch.Tensor, ndim: int) -> torch.Tensor:
    """Performs the operation `sum_k(facs[k]*loads[k])`."""
    if ndim == 1:
        coeffs = coeffs.unsqueeze(2)
    elif ndim == 2:
        coeffs = coeffs.unsqueeze(2).unsqueeze(3)
    elif ndim == 3:
        coeffs = coeffs.unsqueeze(2).unsqueeze(3).unsqueeze(4)
    else: 
        raise NotImplementedError 
    return (coeffs * basis).sum(dim=1).to_dense()
    

class SineFunction1D(object):

    ndim = 1

    def __init__(self, period=1) -> None:
        self.period = period

    def __call__(self, points: torch.Tensor):
        return torch.sin(points * 2 * torch.pi / self.period)
    

class CosineFunction1D(object):

    ndim = 1

    def __init__(self, period=1) -> None:
        self.period = period

    def __call__(self, points: torch.Tensor):
        return torch.cos(points * 2 * torch.pi / self.period)

# TODO: How to handle scalar inputs to Functions? Make everything List[Any]?

class BumpFunction1D(object):

    ndim = 1

    def __init__(
            self, 
            center: float,
            scale: float,
            max: float,
        ) -> None:
        self.center = center
        self.scale = scale
        self.max = max

    def __call__(self, points: torch.Tensor):
        
        # Center then scale points about origin
        points = points.clone()
        points -= torch.tensor(self.center)
        points /= torch.tensor(self.scale)

        # Evaluate transformed points
        vals = torch.zeros_like(points, dtype=torch.float64)
        idx = torch.abs(points) <= 1
        vals[idx] = torch.exp(-1 / (1 - points[idx] ** 2))
        return self.max * math.exp(1) * vals


class BumpFunction2D(object):

    ndim = 2

    def __init__(
            self, 
            center: List[float],
            rotation: float,
            scale: List[float],
            max: float
        ) -> None:
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
        out = self.max * math.exp(1) * vals
        return out
    

class BumpFunction3D(object):

    ndim = 3

    def __init__(
            self, 
            center: List[float],
            rotation: List[float],
            scale: List[float],
            max: float
        ) -> None:
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
# NOTE: All loading functions are defined on [0,1]^D and scaled to unit norm.
    
class LoadingFunction(ABC):
    """Base class for LoadingFunctions which build a loading function of 
    dimension `ndim` from `pieces`. When called, a LoadingFunction realizes
    itself on some grid defined by `points`. Output function will have unit
    norm."""

    def __init__(self):
        if not self._compatible_pieces():
            raise Exception("Pieces do not having matching dimensions!")

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        vals = torch.zeros(len(points), dtype=torch.float64)
        for piece in self.pieces:
            vals += piece(points)
        return safe_normalize(vals)
    
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


class SineLoading1D(LoadingFunction):

    ndim = 1
    pieces = [SineFunction1D(period=1)]


class CosineLoading1D(LoadingFunction):

    ndim = 1
    pieces = [CosineFunction1D(period=1)]


class BumpPairLoading1D1(LoadingFunction):

    ndim = 1
    pieces = [
        BumpFunction1D(0.2, 0.15, 1),
        BumpFunction1D(0.6, 0.15, 1),
    ]


class BumpPairLoading1D2(LoadingFunction):

    ndim = 1
    pieces = [
        BumpFunction1D(0.4, 0.15, 1),
        BumpFunction1D(0.8, 0.15, 1),
    ]


class CornerPairLoading2D1(LoadingFunction):

    ndim = 2
    pieces = [
        BumpFunction2D([0.25, 0.25], 0, [0.15, 0.15], 1),
        BumpFunction2D([0.75, 0.75], 0, [0.15, 0.15], 1),
    ]


class CornerPairLoading2D2(LoadingFunction):

    ndim = 2
    pieces = [
        BumpFunction2D([0.25, 0.75], 0, [0.15, 0.15], 1),
        BumpFunction2D([0.75, 0.25], 0, [0.15, 0.15], 1),
    ]

class CornerPairLoading2D3(LoadingFunction):

    ndim = 2
    pieces = [
        BumpFunction2D([0.25, 0.25], 0, [0.25, 0.25], 1),
        BumpFunction2D([0.75, 0.75], 0, [0.25, 0.25], 1),
    ]


class CornerPairLoading2D4(LoadingFunction):

    ndim = 2
    pieces = [
        BumpFunction2D([0.25, 0.75], 0, [0.25, 0.25], 1),
        BumpFunction2D([0.75, 0.25], 0, [0.25, 0.25], 1),
    ]


class CornerPairLoading3D1(LoadingFunction):

    ndim = 3
    pieces = [
        BumpFunction3D([0.25, 0.25, 0.25], [0, 0, 0], [0.15, 0.15, 0.15], 1),
        BumpFunction3D([0.75, 0.75, 0.75], [0, 0, 0], [0.15, 0.15, 0.15], 1),
    ]


class CornerPairLoading3D2(LoadingFunction):

    ndim = 3
    pieces = [
        BumpFunction3D([0.25, 0.25, 0.75], [0, 0, 0], [0.15, 0.15, 0.15], 1),
        BumpFunction3D([0.75, 0.75, 0.25], [0, 0, 0], [0.15, 0.15, 0.15], 1),
    ]


class CornerPairLoading3D3(LoadingFunction):

    ndim = 3
    pieces = [
        BumpFunction3D([0.25, 0.75, 0.25], [0, 0, 0], [0.15, 0.15, 0.15], 1),
        BumpFunction3D([0.75, 0.25, 0.75], [0, 0, 0], [0.15, 0.15, 0.15], 1),
    ]


class CornerPairLoading3D4(LoadingFunction):

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
    

class TrigLoadingScheme1D1(LoadingScheme):

    ndim = 1
    loading_fcns = [
        SineLoading1D, 
        CosineLoading1D
    ]
    scales = [3, 2]


class BumpLoadingScheme1D1(LoadingScheme):

    ndim = 1
    loading_fcns = [
        BumpPairLoading1D1, 
        BumpPairLoading1D2
    ]
    scales = [2, 1]


class BumpLoadingScheme2D1(LoadingScheme):

    ndim = 2
    loading_fcns = [
        CornerPairLoading2D1, 
        CornerPairLoading2D2
    ]
    scales = [4, 3]

class BumpLoadingScheme2D2(LoadingScheme):

    ndim = 2
    loading_fcns = [
        CornerPairLoading2D3,
        CornerPairLoading2D4
    ]
    scales = [2, 1]
        

class BumpLoadingScheme3D1(LoadingScheme):

    ndim = 3
    loading_fcns = [
        CornerPairLoading3D1, 
        CornerPairLoading3D2,
        CornerPairLoading3D3,
        CornerPairLoading3D4
    ]
    scales = [2, 2, 2, 2]


LOADING_SCHEMES = {
    'TrigScheme1D1': TrigLoadingScheme1D1,
    'BumpScheme1D1': BumpLoadingScheme1D1,
    'BumpScheme2D1': BumpLoadingScheme2D1,
    'BumpScheme2D2': BumpLoadingScheme2D2,
    'BumpScheme3D1': BumpLoadingScheme3D1,
}


# ---------- ERROR FUNCTIONS ---------- #
# NOTE: All error functions are defined on [0,1]^D and scaled to unit norm.

class ErrorFunction(ABC):
    """Base class for ErrorFunctions which build an error function of 
    dimension `ndim`. On instantiation, ErrorFunctions instantiate some
    `fcn` which is then evaluated then normalized within the class' __call__
    method."""

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        vals = self.fcn(points)
        return safe_normalize(vals)


class BumpErrorFunction1D1(ErrorFunction):

    ndim = 1

    def __init__(self, center: float) -> None:
        self.fcn = BumpFunction1D(center, 0.05, 1)


class BumpErrorFunction2D1(ErrorFunction):

    ndim = 2

    def __init__(self, center: List[float]) -> None:
        self.fcn = BumpFunction2D(center, 0, [0.05, 0.05], 1)


class BumpErrorFunction3D1(ErrorFunction):

    ndim = 3

    def __init__(self, center: List[float]) -> None:
        self.fcn = BumpFunction3D(center, [0, 0, 0], [0.05, 0.05, 0.05], 1)


# ---------- ERROR SCHEMES ---------- #

class ErrorScheme(ABC):

    def __init__(self, gen: torch.Generator) -> None:
        
        # Check dimension compatibility
        if self.ndim != self.error_fcn.ndim: 
            raise Exception("Dimension of `error_fcn` does not match `ndim`!")
        
        # Generate centers and scales for each error function
        self.centers = torch.rand(self.ncomps, self.ndim, generator=gen, dtype=torch.float64)
        self.scales = torch.rand(self.ncomps, generator=gen, dtype=torch.float64)
        self.scales *= (self.scale_max - self.scale_min)
        self.scales += self.scale_min

    def __call__(self, points: torch.Tensor) -> torch.sparse.Tensor:
        
        # Compile length-ncomps list for indices and values
        indices_list = [None] * self.ncomps
        values_list = [None] * self.ncomps
        for j in range(self.ncomps):
            out = self.error_fcn(self.centers[j].tolist())(points)
            nz_idx = torch.nonzero(out).t()[0]
            indices_list[j] = torch.row_stack((  # 2-by-len(nz_idx) tensor --> [[j, ..., j], [#, ..., #]]
                j * torch.ones(len(nz_idx), dtype=torch.int32),
                nz_idx
            ))
            values_list[j] = self.scales[j] * out[nz_idx]

        # Return sparse tensor
        indices = torch.cat(indices_list, dim=1)
        values = torch.cat(values_list)
        sz = [self.ncomps, len(points)]
        return torch.sparse_coo_tensor(indices, values, sz)
        

    @property
    @abstractmethod
    def ndim(self):
        pass

    @property
    @abstractmethod
    def error_fcn(self):
        pass

    @property
    @abstractmethod
    def ncomps(self):
        pass

    @property
    @abstractmethod
    def scale_min(self):
        pass

    @property
    @abstractmethod
    def scale_max(self):
        pass


class BumpErrorScheme1D1(ErrorScheme):

    ndim = 1
    error_fcn = BumpErrorFunction1D1
    ncomps = 100
    scale_min = 0
    scale_max = 1

    def __init__(self, gen: torch.Generator) -> None:
        super().__init__(gen)


class BumpErrorScheme2D1(ErrorScheme):

    ndim = 2
    error_fcn = BumpErrorFunction2D1
    ncomps = 1000
    scale_min = 0
    scale_max = 1

    def __init__(self, gen: torch.Generator) -> None:
        super().__init__(gen)


class BumpErrorScheme3D1(ErrorScheme):

    ndim = 3
    error_fcn = BumpErrorFunction3D1
    ncomps = 10000
    scale_min = 0
    scale_max = 1

    def __init__(self, gen: torch.Generator) -> None:
        super().__init__(gen)


ERROR_SCHEMES = {
    'BumpScheme1D1': BumpErrorScheme1D1,
    'BumpScheme2D1': BumpErrorScheme2D1,
    'BumpScheme3D1': BumpErrorScheme3D1,
}
 


# ---------- MAIN FUNCTIONS ---------- #

def build_loadings(
        indices: torch.Tensor, 
        grid_shape: List[int], 
        vals: torch.Tensor
    ) -> torch.Tensor:
    """Returns a (dense) loading tensor of dimension ncomps-by-grid_shape."""

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


def build_errors(
        indices: torch.Tensor,
        grid_shape: List[int],
        vals: torch.sparse.Tensor
    ) -> torch.sparse.Tensor:
    """Returns a (sparse) error tensor of dimension ncomps-by-grid_shape."""
    
    # NOTE: `vals` is a sparse tensor of dimension ncomps-by-n, where n is the 
    # number of points in the grid. To obtain the desired sparse tensor, we 
    # need only "unfold" the dimension of size n into several whose sizes are
    # specified in `grid_shape`. We can do this my mapping the ith point in the
    # grid to the ith element in `indices`.
    ncomps = vals.shape[0]
    indices_ = vals._indices()
    idx = indices[indices_[1]].t()  # indices_[1] contains flattened point indices
    idx = torch.row_stack((indices_[0], idx))  # indices_[0] contains components
    size = [ncomps, *grid_shape]
    return torch.sparse_coo_tensor(
        indices=idx, 
        values=vals._values(), 
        size=size
    )



def simulate_ffm_data(
        loads: torch.Tensor,
        err: torch.sparse.Tensor, 
        num_samps: int,
        batch_size: int,
        gen: torch.Generator = torch.Generator(),
    ):
    num_facs = loads.shape[0]
    grid_shape = loads.shape[1:]
    ndim = len(grid_shape)
    num_err_comps = err.shape[0]
    while num_samps > 0:
        n_batch = min(batch_size, num_samps) 
        facs = torch.normal(0, 1, (n_batch, num_facs), dtype=torch.float64, generator=gen)
        err_coeffs = torch.normal(0, 1, (n_batch, num_err_comps), dtype=torch.float64, generator=gen)
        data = (
            nsamp_basis_comb(loads, facs, ndim) + 
            nsamp_basis_comb(err, err_coeffs, ndim)
        )
        yield data
        num_samps -= n_batch


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str,
        help="Configuration (i.e., mode) in which to run script."
    )
    parser.add_argument(
        '--dir',
        help="Dataset directory in which simulated data will be stored."
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
        '--err_scheme', type=str,
        help="Error scheme used to simulate data."
    )
    parser.add_argument(
        '--num_samps', type=int,
        help="Number of training samples to simulate."
    )
    parser.add_argument(
        '--batch_size', type=int,
        help="Maximum number of samples per output file."
    )
    parser.add_argument(
        '--seed', default=12345, type=int,
        help="Integer used to seed generator."
    )
    args = parser.parse_args()

    # Configure globals
    config = load_config(args.config)
    gen = torch.Generator().manual_seed(args.seed)
    load_scheme = LOADING_SCHEMES[args.load_scheme]()
    err_scheme = ERROR_SCHEMES[args.err_scheme](gen)

    # Check for `load_scheme`, `err_scheme`, and `grid_shape` compatibility
    if load_scheme.ndim != len(args.grid_shape):
        msg = ("Number of loading scheme dimensions does not match the number " 
               "of grid dimensions!")
        raise Exception(msg)       
    if err_scheme.ndim != len(args.grid_shape):
        msg = ("Number of error scheme dimensions does not match the number " 
               "of grid dimensions!")
        raise Exception(msg)  


    # ---------- LOADING AND ERROR PREP ---------- #

    print("Preparing loadings and errors...")

    # Delete files from directory
    dir_out = os.path.join(config.scratch_root, 'datasets', args.dir)
    refresh_directory(dir_out)

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

    # Build error tensor
    vals = err_scheme(points)  # ncomps-by-n (sparse)
    errs = build_errors(indices, args.grid_shape, vals)


    # ---------- DATA SIMULATION ---------- #

    print("Simulating new data...")
    dataloader = simulate_ffm_data(
        loads, 
        errs,
        num_samps=args.num_samps,
        batch_size=args.batch_size,
        gen=gen
    )
    write_generated_tensor(dataloader, dir_out, 'data')

    print("DONE!")
