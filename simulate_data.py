import argparse
import os
import math
import numpy as np
import torch
from abc import ABC, abstractmethod
from functools import partial
from scipy.interpolate import BSpline
from skfda.representation.basis import Basis, BSplineBasis, TensorBasis
from typing import Callable, List, Union

from config import load_config
from utils.utils import (
    gen_seeds,
    refresh_directory, 
    safe_l2_normalization, 
    slice_sparse_coo_tensor,
    write_generated_tensor
)


# ==================== UTILITIES ==================== #

def basis_expansion(basis: Union[torch.Tensor], coeffs: torch.Tensor, ndim: int) -> torch.Tensor:
    """Performs the operation `sum_k(coeffs[k]*loads[k])`."""
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
    
    
def bspline_basis_fcns(
        n_basis: int, 
        domain_range: List[float], 
        order: int
    ) -> List[Callable]:

    # Define the knot sequence, then pad
    knots = np.linspace(domain_range[0], domain_range[1], n_basis - order + 1)
    knots = np.concatenate(([domain_range[0]] * order, knots, [domain_range[1]] * order))

    # Get list of basis functions
    fcns = []
    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        basis = BSpline(knots, coeffs, order)
        fcns.append(basis)
    
    return fcns
    

# ==================== FACTORS ==================== #

def squared_exponential_kernel(dist, length):
    return torch.exp(-dist ** 2 / 2 / length ** 2)


def build_factors(
        grid: torch.Tensor, 
        num_facs: int, 
        kernel: Callable, 
        gen: torch.Generator
    ):
    """Simulate `num_facs` functional factors from a Gaussian Process defined
    by `kernel`.

    Args:
        grid (torch.Tensor): Temporal grid on which to simulate factors.
        num_facs (int): Number of factors to simulate.
        kernel (Callable): Kernel defining the GP.
        gen (torch.Generator): PyTorch generator.

    Returns:
        torch.Tensor: A `sz_time`-by-`num_facs` tensor of factors.
    """

    # NOTE: Annoyingly, PyTorch does not allow you to sample from a MVN using
    # a Generator. Thankfully, NumPy does, so we use its function. To do so, 
    # we need to get a NumPy generator from a PyTorch generator. 
    gen = np.random.default_rng(gen_seeds(gen, 1))

    facs = torch.zeros((num_facs, len(grid)))
    mean = torch.zeros(len(grid))
    temp1, temp2 = torch.meshgrid(grid, grid, indexing='ij')
    dists = torch.abs(temp1 - temp2)
    cov = kernel(dists)
    for k in range(num_facs):
        temp = gen.multivariate_normal(mean.numpy(), cov.numpy())  # convert tensors to ndarrays
        facs[k] = torch.tensor(temp)
    return facs.t()


FACTOR_KERNELS = {
    'SqExp100': partial(squared_exponential_kernel, length=0.100),  # Smooth
    'SqExp050': partial(squared_exponential_kernel, length=0.050),
    'SqExp040': partial(squared_exponential_kernel, length=0.040),
    'SqExp030': partial(squared_exponential_kernel, length=0.030),
    'SqExp020': partial(squared_exponential_kernel, length=0.020),
    'SqExp010': partial(squared_exponential_kernel, length=0.010),
    'SqExp001': partial(squared_exponential_kernel, length=0.001)   # Rough
}


# ==================== LOADINGS ==================== #

# -------------------- Loading Functions -------------------- #
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
        return safe_l2_normalization(vals)
    
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
        BumpFunction2D([0.25, 0.25], 0, [0.25, 0.25], 1),
        BumpFunction2D([0.75, 0.75], 0, [0.25, 0.25], 1),
    ]


class CornerPairLoading2D2(LoadingFunction):

    ndim = 2
    pieces = [
        BumpFunction2D([0.25, 0.75], 0, [0.25, 0.25], 1),
        BumpFunction2D([0.75, 0.25], 0, [0.25, 0.25], 1),
    ]


class EdgePairLoading2D1(LoadingFunction):

    ndim = 2
    pieces = [
        BumpFunction2D([0.25, 0.5], 0, [0.25, 0.25], 1),
        BumpFunction2D([0.75, 0.5], 0, [0.25, 0.25], 1),
    ]


class EdgePairLoading2D2(LoadingFunction):

    ndim = 2
    pieces = [
        BumpFunction2D([0.5, 0.75], 0, [0.25, 0.25], 1),
        BumpFunction2D([0.5, 0.25], 0, [0.25, 0.25], 1),
    ]


class DefaultNetLoading2D(LoadingFunction):

    ndim = 2
    pieces = [
        BumpFunction2D(center=[0.5, 0.25], rotation=0, scale=[0.1, 0.15], max=1),
        BumpFunction2D(center=[0.3, 0.1], rotation=30, scale=[0.05, 0.1], max=1),
        BumpFunction2D(center=[0.7, 0.1], rotation=-30, scale=[0.05, 0.1], max=1),
        BumpFunction2D(center=[0.5, 0.9], rotation=0, scale=[0.1, 0.05], max=0.5),
        BumpFunction2D(center=[0.4, 0.8], rotation=0, scale=[0.05, 0.05], max=0.5),
        BumpFunction2D(center=[0.6, 0.8], rotation=0, scale=[0.05, 0.05], max=0.5),
    ]


class ExecutiveNetLoading2D(LoadingFunction):

    ndim = 2
    pieces = [
        BumpFunction2D(center=[0.5, 0.8], rotation=0, scale=[0.1, 0.15], max=1),
        BumpFunction2D(center=[0.4, 0.8], rotation=45, scale=[0.15, 0.1], max=1),
        BumpFunction2D(center=[0.6, 0.8], rotation=-45, scale=[0.15, 0.1], max=1),
        BumpFunction2D(center=[0.5, 0.5], rotation=0, scale=[0.05, 0.05], max=0.5),
        BumpFunction2D(center=[0.8, 0.35], rotation=0, scale=[0.05, 0.05], max=0.5),
        BumpFunction2D(center=[0.2, 0.35], rotation=0, scale=[0.05, 0.05], max=0.5),
        BumpFunction2D(center=[0.55, 0.1], rotation=0, scale=[0.05, 0.05], max=0.5),
        BumpFunction2D(center=[0.45, 0.1], rotation=0, scale=[0.05, 0.05], max=0.5),
    ]


class RightVisualNetLoading2D(LoadingFunction):

    ndim = 2
    pieces = [
        BumpFunction2D(center=[0.3, 0.25], rotation=30, scale=[0.15, 0.25], max=1),
        BumpFunction2D(center=[0.35, 0.8], rotation=-30, scale=[0.15, 0.25], max=1),
        BumpFunction2D(center=[0.7, 0.25], rotation=30, scale=[0.15, 0.1], max=0.7),
    ]


class LeftVisualNetLoading2D(LoadingFunction):

    ndim = 2
    pieces = [
        BumpFunction2D(center=[0.7, 0.25], rotation=-30, scale=[0.15, 0.25], max=1),
        BumpFunction2D(center=[0.65, 0.8], rotation=30, scale=[0.15, 0.25], max=1),
        BumpFunction2D(center=[0.3, 0.25], rotation=-30, scale=[0.15, 0.1], max=0.7),
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


# -------------------- Loading Schemes -------------------- #
    
class LoadingScheme(ABC):
    """Base class for LoadingSchemes which use `loading_fcns` and `scales` to 
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
    

class TrigLoadingScheme1D2K(LoadingScheme):

    ndim = 1
    loading_fcns = [
        SineLoading1D, 
        CosineLoading1D
    ]
    scales = [1, 1]


class BumpLoadingScheme1D2K(LoadingScheme):

    ndim = 1
    loading_fcns = [
        BumpPairLoading1D1, 
        BumpPairLoading1D2
    ]
    scales = [1, 1]


class BumpLoadingScheme2D2K(LoadingScheme):

    ndim = 2
    loading_fcns = [
        CornerPairLoading2D1, 
        CornerPairLoading2D2
    ]
    scales = [1, 1]


class BumpLoadingScheme2D4K(LoadingScheme):

    ndim = 2
    loading_fcns = [
        CornerPairLoading2D1, 
        CornerPairLoading2D2,
        EdgePairLoading2D1,
        EdgePairLoading2D2
    ]
    scales = [1, 1, 1, 1]


class NetLoadingScheme2D2K(LoadingScheme):

    ndim = 2
    loading_fcns = [
        DefaultNetLoading2D,
        ExecutiveNetLoading2D,
    ]
    scales = [1, 1]


class NetLoadingScheme2D4K(LoadingScheme):

    ndim = 2
    loading_fcns = [
        DefaultNetLoading2D,
        ExecutiveNetLoading2D,
        RightVisualNetLoading2D,
        LeftVisualNetLoading2D,
    ]
    scales = [1, 1, 1, 1]
        

class BumpLoadingScheme3D4K(LoadingScheme):

    ndim = 3
    loading_fcns = [
        CornerPairLoading3D1, 
        CornerPairLoading3D2,
        CornerPairLoading3D3,
        CornerPairLoading3D4
    ]
    scales = [1, 1, 1, 1]


LOADING_SCHEMES = {
    'TrigScheme1D2K': TrigLoadingScheme1D2K,
    'BumpScheme1D2K': BumpLoadingScheme1D2K,
    'BumpScheme2D2K': BumpLoadingScheme2D2K,
    'BumpScheme2D4K': BumpLoadingScheme2D4K,
    'BumpScheme3D4K': BumpLoadingScheme3D4K,
}



# ==================== ERRORS ==================== #

# ---------- Bases ---------- #

class BSplinePinnedNoExtrapBasis(Basis):

    def __init__(
            self,
            domain_range: List[float],
            n_basis: int,
            order: int
        ) -> None:
        super().__init__(domain_range=domain_range, n_basis=n_basis)
        self.fcns = bspline_basis_fcns(n_basis + 2, domain_range, order)
        self.fcns = self.fcns[1:(n_basis + 1)]  # exclude boundary functions

    def _evaluate(
            self,
            eval_points: np.ndarray
        ) -> np.ndarray:
        eval_points = eval_points[..., 0]
        mask1 = eval_points >= self._domain_range[0][0]
        mask2 = eval_points <= self._domain_range[0][1]
        mask = np.logical_and(mask1, mask2)
        out = np.zeros((len(self.fcns), len(eval_points)))
        out[:,mask] = np.vstack([f(eval_points[mask]) for f in self.fcns])
        return out
    

# ---------- Function Sets ---------- #
# NOTE: Although SpatialErrorFcnSets and TemporalErrorFcnSets are not 
# technically bases as we do not require them to be linearly independent, we 
# want to leverage skfda's Basis framework, so these classes inherit from Basis. 

class SpatialErrorBump1DFcnSet(Basis):

    def __init__(
            self, 
            domain_range: List[float],
            n_basis: int, 
            width: float
        ) -> None:
        super().__init__(domain_range=domain_range, n_basis=n_basis)
        centers = torch.linspace(0, 1, n_basis, dtype=torch.float64)
        self.fcns = [BumpFunction1D(c.item(), width / 2, 1) for c in centers]

    def _evaluate(
            self, 
            eval_points: np.ndarray
        ) -> np.ndarray:
        eval_points = torch.tensor(eval_points[..., 0], dtype=torch.float64)
        out = torch.zeros((self._n_basis, len(eval_points)), dtype=torch.float64)
        for j in range(self._n_basis):
            out[j] = self.fcns[j](eval_points)
        return out.numpy()


class SpatialErrorBSplinePinned1DFcnSet(Basis):

    def __init__(
            self,
            domain_range: List[float],
            n_basis: int,
            width: float,
            gen: np.random.Generator  
        ) -> None:
        super().__init__(domain_range=domain_range, n_basis=n_basis)

        # Create n_basis "local bases" from which this set's functions will 
        # be generated
        n_basis_base = 5
        self.local_bases = [None] * n_basis

        left_endpoints = np.linspace(
            start=domain_range[0] - width/2, 
            stop=domain_range[1] - width/2, 
            num=n_basis
        )
        for j in range(n_basis):
            self.local_bases[j] = BSplinePinnedNoExtrapBasis(
                [left_endpoints[j], left_endpoints[j] + width], 
                n_basis_base, 4
            )

        # Generate coefficients that will be used generate single functions 
        # from local bases
        self.coeffs = gen.normal(size=(n_basis, n_basis_base))

    def _evaluate(
            self,
            eval_points: np.ndarray
        ) -> np.ndarray:
        out = np.zeros((self._n_basis, len(eval_points)))
        for j in range(self._n_basis):
            temp = self.local_bases[j](eval_points)[:,:,0]
            out[j] = np.matmul(self.coeffs[j], temp)
        return out


class TemporalErrorBSplineFcnSet(Basis):

    def __init__(
            self,
            domain_range: List[float],
            n_basis: int,
            n_basis_base: int,
            gen: np.random.Generator  
        ) -> None:
        super().__init__(domain_range=domain_range, n_basis=n_basis)

        # Create a basis from which functions within this set will be generated
        self.basis_base = BSplineBasis([0, 1], n_basis=n_basis_base, order=4)
        
        # Generate coefficients that will be used generate single functions
        # of this set
        self.coeffs = gen.normal(size=(n_basis, n_basis_base))

    def _evaluate(
            self,
            eval_points: np.ndarray
        ) -> None:
        out = np.zeros((self._n_basis, len(eval_points)))
        vals = self.basis_base(eval_points)[:,:,0]
        for j in range(self._n_basis):
            out[j] = np.matmul(self.coeffs[j], vals)
        return out


# ---------- Error Schemes ---------- #
# NOTE: Error schemes are defined by a `fcn_set`, and a range 
# (`scale_min`, `scale_mix`) from which to draw `n_fcns` scaling factors. 

class ErrorScheme(ABC):
        
    def __init__(
            self, 
            width_space: float,
            gen: torch.Generator
        ) -> None:

        # Construct function set
        gen_np = np.random.default_rng(gen_seeds(gen, 1))
        self.fcn_set = self.get_fcn_set(width_space, gen_np)
        self.n_fcns = len(self.fcn_set)
        self.ndim = len(self.fcn_set.basis_list)

        # Generate coefficients for functions in tensor product set
        self.coeffs = torch.rand(self.n_fcns, generator=gen, dtype=torch.float64)
        self.coeffs *= (self.scale_max - self.scale_min)
        self.coeffs += self.scale_min

    def __call__(
            self, 
            points: torch.Tensor, 
            batch_size: int = 1000
        ) -> torch.Tensor:

        # Build sparse tensor of points in batches
        indices_list = []
        values_list = []
        n_points = len(points)
        start_idx = 0
        while start_idx < n_points:

            # Evaluate batch of points
            sz = min(batch_size, n_points - start_idx)
            points_ = points[start_idx:(start_idx + sz)]
            vals = torch.tensor(self.fcn_set(points_))[..., 0]  # n_fcns-by-sz

            # Append nonzero indices/values
            nz_idx = torch.nonzero(vals).t()  # 2-by-`num nonzero values`
            values_list.append(vals[nz_idx[0], nz_idx[1]])
            nz_idx[1] += start_idx  # shift point indices to acocunt for batching
            indices_list.append(nz_idx)

            # Update start_idx
            start_idx += sz

        # Create sparse tensor
        indices = torch.cat(indices_list, dim=1)
        values = torch.cat(values_list)
        sz = [self.n_fcns, len(points)]
        out = torch.sparse_coo_tensor(indices, values, sz)

        # Return normalized-then-scaled sparse tensor
        # NOTE: If batching, normalizing within the spatial/temporal function
        # sets and the combined function set will lead to inconsistencies 
        # between batch sizes. So only normalize here. 
        new_row_indices = [None] * self.n_fcns
        new_col_indices = [None] * self.n_fcns
        new_values = [None] * self.n_fcns
        for j in range(self.n_fcns):

            # Normalize values in jth row
            new_vals = safe_l2_normalization(out[j].to_dense())
            new_vals *= self.coeffs[j]
            nz_idx = torch.nonzero(new_vals)
            new_values[j] = new_vals[nz_idx]

            # Collect row/column indices
            col_idx = out[j].coalesce().indices()[0]
            row_idx = j * torch.ones(len(col_idx), dtype=torch.int32)
            new_col_indices[j] = col_idx
            new_row_indices[j] = row_idx
        
        return torch.sparse_coo_tensor(
            indices=torch.row_stack([
                torch.cat(new_row_indices), 
                torch.cat(new_col_indices)
            ]),
            values=torch.cat(new_values).t()[0],
            size=sz
        )

    @abstractmethod
    def get_fcn_set(self, width_space: float, gen_np: np.random.Generator) -> Basis:
        pass

    @property
    @abstractmethod
    def scale_min(self):
        pass

    @property
    @abstractmethod
    def scale_max(self):
        pass


# NOTE: Naming convention for error schemes is as follows:
#           <temporal-fset>_<spatial-fset><ndim_space>D_ErrorScheme
    
class BSpline_Bump1D_ErrorScheme(ErrorScheme):

    scale_min = 0.1
    scale_max = 1

    def get_fcn_set(self, width_space, gen_np):
        fcn_set_time = TemporalErrorBSplineFcnSet(
            domain_range=[0, 1], 
            n_basis=10, 
            n_basis_base=40,
            gen=gen_np
        )
        fcn_set_space = SpatialErrorBump1DFcnSet(
            domain_range=[0, 1], 
            n_basis=20, 
            width=width_space
        )
        return TensorBasis([
            fcn_set_time, 
            fcn_set_space
        ])
    

class BSpline_BSplinePinned1D_ErrorScheme(ErrorScheme):

    scale_min = 0.1
    scale_max = 1

    def get_fcn_set(self, width_space, gen_np):
        fcn_set_time = TemporalErrorBSplineFcnSet(
            domain_range=[0, 1], 
            n_basis=10, 
            n_basis_base=40,
            gen=gen_np
        )
        fcn_set_space = SpatialErrorBSplinePinned1DFcnSet(
            domain_range=[0, 1], 
            n_basis=20, 
            width=width_space, 
            gen=gen_np
        )
        return TensorBasis([
            fcn_set_time, 
            fcn_set_space
        ])
    

class BSpline_Bump2D_ErrorScheme(ErrorScheme):

    scale_min = 0.1
    scale_max = 1

    def get_fcn_set(self, width_space, gen_np):
        fcn_set_time = TemporalErrorBSplineFcnSet(
            domain_range=[0, 1], 
            n_basis=10, 
            n_basis_base=40,
            gen=gen_np
        )
        fcn_set_space = SpatialErrorBump1DFcnSet(
            domain_range=[0, 1], 
            n_basis=20, 
            width=width_space
        )
        return TensorBasis([
            fcn_set_time, 
            fcn_set_space,
            fcn_set_space
        ])
    

class BSpline_BSplinePinned2D_ErrorScheme(ErrorScheme):

    scale_min = 0.1
    scale_max = 1

    def get_fcn_set(self, width_space, gen_np):
        fcn_set_time = TemporalErrorBSplineFcnSet(
            domain_range=[0, 1], 
            n_basis=10, 
            n_basis_base=40,
            gen=gen_np
        )
        fcn_set_space = SpatialErrorBSplinePinned1DFcnSet(
            domain_range=[0, 1], 
            n_basis=20, 
            width=width_space, 
            gen=gen_np
        )
        return TensorBasis([
            fcn_set_time, 
            fcn_set_space, 
            fcn_set_space
        ])
    

ERROR_SCHEMES = {

    # 1-dimensional
    'BSpline_Bump1D': BSpline_Bump1D_ErrorScheme,
    'BSpline_BSplinePinned1D': BSpline_BSplinePinned1D_ErrorScheme,

    # 2-dimensinoal
    'BSpline_Bump2D': BSpline_Bump2D_ErrorScheme,
    'BSpline_BSplinePinned2D': BSpline_BSplinePinned2D_ErrorScheme

}
 


# ---------- MAIN FUNCTIONS ---------- #

def build_loadings(
        indices: torch.Tensor, 
        sz_space: List[int], 
        vals: torch.Tensor
    ) -> torch.Tensor:
    """Returns a (dense) loading tensor of dimension ncomps-by-sz_space."""

    ndim = len(sz_space)
    ncomps = vals.shape[0]
    loads = torch.zeros(ncomps, *sz_space, dtype=torch.float64)

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
        sz: List[int],
        vals: torch.sparse.Tensor
    ) -> torch.sparse.Tensor:
    """Returns a (sparse) error tensor of dimension ncomps-by-sz_time-bysz_space."""
    
    # NOTE: `vals` is a sparse tensor of dimension ncomps-by-n, where n is the 
    # number of points in the spatiotemproal grid. To obtain the desired sparse tensor, we 
    # need only "unfold" the dimension of size n into several whose sizes are
    # specified in `sz_space`. We can do this my mapping the ith point in the
    # grid to the ith element in `indices`.
    ncomps = vals.size(0)
    indices_ = vals._indices()
    idx = indices[indices_[1]].t()  # indices_[1] contains flattened point indices
    idx = torch.row_stack((indices_[0], idx))  # indices_[0] contains components
    sz = [ncomps] + sz
    return torch.sparse_coo_tensor(
        indices=idx, 
        values=vals._values(), 
        size=sz
    )


def simulate_ffm_data(
        loads: torch.Tensor,
        facs: torch.Tensor,
        err: torch.sparse.Tensor, 
        prop_global: float,
        batch_size: int,
        gen: torch.Generator = torch.Generator(),
    ):

    def _gen_global_local_batches():
        curr_time = 0
        while curr_time < sz_time: 
            n_batch = min(batch_size, sz_time - curr_time) 
            facs_batch = facs[curr_time:(curr_time + n_batch)]
            comp_global = basis_expansion(loads, facs_batch, ndim_space)
            comp_local = torch.sum(
                slice_sparse_coo_tensor(
                    err, dim=1, 
                    start=curr_time, 
                    stop=curr_time + n_batch
                ),
                dim=0
            )
            yield comp_global, comp_local
            curr_time += n_batch

    # Extract constants
    sz_space = loads.shape[1:]
    sz_time = facs.shape[0]
    ndim_space = len(sz_space)
    n_err_fcns = err.shape[0]

    # Scale error functions
    err_coeffs = torch.normal(0, 1, (n_err_fcns,), dtype=torch.float64, generator=gen)
    err *= err_coeffs.view(n_err_fcns, 1, *[1]*ndim_space)

    # Compute global and local l2 norms
    global_frob = 0
    local_frob = 0
    global_numel = 0
    local_numel = 0
    for comp_global, comp_local in _gen_global_local_batches():
        global_frob += torch.sum(comp_global ** 2).item()
        global_numel += torch.numel(comp_global)
        local_frob += torch.sum(comp_local ** 2).item()
        local_numel += torch.numel(comp_local)
    norm_global = math.sqrt(global_frob / global_numel)
    norm_local = math.sqrt(local_frob / local_numel)


    # Yield data in desired global-to-local ratio
    for comp_global, comp_local in _gen_global_local_batches():
        comp_global /= norm_global
        comp_global *= prop_global
        comp_local /= norm_local
        comp_local *= (1 - prop_global)
        yield comp_global + comp_local



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
        '--sz_time', type=int,
        help=("Number of time points for which to simulate data. Since the "
              "temporal domain is 1-dimensional, this an integer argument.")
    )
    parser.add_argument(
        '--sz_space', nargs='+', type=int,
        help="Shape of the spatial grid on which to simulate data."
    )
    parser.add_argument(
        '--factor_kernel', type=str,
        help="Kernel to use when simulating factors from MVN."
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
        '--delta', type=float,
        help="Bandwidth of spatial error covariance."
    )
    parser.add_argument(
        '--prop_global', type=float,
        help="Proportion of observations coming from global component."
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
    err_scheme = ERROR_SCHEMES[args.err_scheme](args.delta, gen)
    sz = [args.sz_time] + args.sz_space

    # Check for `load_scheme`, `err_scheme`, and `sz_space` compatibility
    if load_scheme.ndim != len(args.sz_space):
        msg = ("Number of loading scheme dimensions does not match the number " 
               "of spatial dimensions!")
        raise Exception(msg)       
    if err_scheme.ndim != len(args.sz_space) + 1:
        msg = ("Number of error scheme dimensions does not match the number " 
               "of spatiotemporal dimensions!")
        raise Exception(msg)  


    # ---------- LOADING, FACTOR, AND ERROR PREP ---------- #

    print("Preparing indices and points...")

    # Delete files from directory
    refresh_directory(args.dir)

    # Generate `points_space` and `indices_space` from `sz_space`
    points_space_list = []
    indices_space_list = []
    for sz_ in args.sz_space:
        indices_ = torch.arange(sz_, dtype=torch.int32)
        points_ = indices_.to(torch.float64) / sz_
        indices_space_list.append(indices_)
        points_space_list.append(points_)
    indices_space = torch.cartesian_prod(*indices_space_list)  # M-by-ndim_space
    points_space = torch.cartesian_prod(*points_space_list)    # M-by-ndim_space

    # Generate `points_time` and `indices_time` from `sz_time`
    indices_time = torch.arange(args.sz_time, dtype=torch.int32)
    points_time = indices_time.to(torch.float64) / args.sz_time

    # Get `points` and `indices` by taking cartesian product
    indices = torch.cartesian_prod(indices_time, *indices_space_list)
    points = torch.cartesian_prod(points_time, *points_space_list)

    # Build loading tensor
    print("Preparing loadings...")
    vals = load_scheme(points_space)  # n_facs-by-sz_shape
    loads = build_loadings(indices_space, args.sz_space, vals)

    # Build factor tensor
    print("Preparing factors...")
    num_facs = len(load_scheme.loading_fcns)
    kernel = FACTOR_KERNELS[args.factor_kernel]
    facs = build_factors(points_time, num_facs, kernel, gen)  # sz_time-by-n_facs

    # Build error tensor
    # TODO: Is there a way to accelerate error generation? Currenlty much slower
    # than loading/factor generation. Cache error schemes?
    print("Preparing errors....")
    vals = err_scheme(points)  # n_fcns-by-sz_time-by-sz_shape (sparse)
    errs = build_errors(indices, sz, vals)


    # ---------- DATA SIMULATION ---------- #

    print("Simulating new data...")
    dataloader = simulate_ffm_data(
        loads, 
        facs,
        errs,
        args.prop_global,
        batch_size=args.batch_size,
        gen=gen
    )
    write_generated_tensor(dataloader, args.dir, 'data-full')

    print("DONE!")

    # NOTE: It takes 90 seconds to generate data for a 30-by-30 spatial grid on
    # 500 time points! Consider caching error schemes. 
