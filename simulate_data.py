import argparse
import itertools
import math
import numpy as np
import os
import torch
from abc import ABC, abstractmethod
from functools import partial
from scipy.interpolate import BSpline, splrep
from typing import Callable, Generator, List, Sequence, Union

from config import load_config
from utils import (
    gen_seeds,
    multiply_list,
    refresh_directory, 
    reshape_sparse_coo_tensor,
    safe_l2_normalization, 
    slice_sparse_coo_tensor,
    write_generated_tensor
)

DomainRange = Union[
    Sequence[float],
    Sequence[Sequence[float]]
]


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


def squared_exponential_kernel(dist, length):
    return torch.exp(-dist ** 2 / 2 / length ** 2)


def simulate_gauss_procs(
        grid: torch.Tensor, 
        n_procs: int, 
        kernel: Callable, 
        gen: torch.Generator,
        gamma: float = 1e-5
    ):

    # Get covariance
    temp1, temp2 = torch.meshgrid(grid, grid, indexing='ij')
    dists = torch.abs(temp1 - temp2)
    cov = kernel(dists)

    # Generate GPs    
    l = torch.linalg.cholesky(cov + gamma * np.eye(len(grid)))
    z = torch.randn((n_procs, len(grid)), generator=gen, dtype=torch.float64)
    return z @ l.t()



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

    def __init__(self, n_fcns: int) -> None:

        if n_fcns > len(self.loading_fcns):
            raise Exception(("Too many loading functions requested! "
                             f"There are {len(self.loading_fcns)} available."))
        if not self._compatible_fcns():
            raise Exception("Loading functions do not having matching dimensions!")
        if not len(self.loading_fcns) == len(self.scales):
            raise Exception("Number of loading functions differs from number of scales!")
        
        self.n_fcns = n_fcns

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        vals = torch.zeros(self.n_fcns, len(points), dtype=torch.float64)
        for k in range(self.n_fcns):
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

    def _compatible_fcns(self):
        ndims = []
        for fcn in self.loading_fcns:
            ndims.append(fcn.ndim)
        return all(d == self.ndim for d in ndims)
    

class TrigLoadingScheme1D(LoadingScheme):

    ndim = 1
    loading_fcns = [
        SineLoading1D, 
        CosineLoading1D
    ]
    scales = [1, 1]


class BumpLoadingScheme1D(LoadingScheme):

    ndim = 1
    loading_fcns = [
        BumpPairLoading1D1, 
        BumpPairLoading1D2
    ]
    scales = [1, 1]


class BumpLoadingScheme2D(LoadingScheme):

    ndim = 2
    loading_fcns = [
        CornerPairLoading2D1, 
        CornerPairLoading2D2,
        EdgePairLoading2D1,
        EdgePairLoading2D2
    ]
    scales = [1, 1, 1, 1]


class NetLoadingScheme2D(LoadingScheme):

    ndim = 2
    loading_fcns = [
        DefaultNetLoading2D,
        ExecutiveNetLoading2D,
        RightVisualNetLoading2D,
        LeftVisualNetLoading2D,
    ]
    scales = [1, 1, 1, 1]
        

class BumpLoadingScheme3D(LoadingScheme):

    ndim = 3
    loading_fcns = [
        CornerPairLoading3D1, 
        CornerPairLoading3D2,
        CornerPairLoading3D3,
        CornerPairLoading3D4
    ]
    scales = [1, 1, 1, 1]


LOADING_SCHEMES = {

    'TrigScheme1D': TrigLoadingScheme1D,

    'BumpScheme1D': BumpLoadingScheme1D,
    'BumpScheme2D': BumpLoadingScheme2D,
    'BumpScheme3D': BumpLoadingScheme3D,

    'NetScheme2D': NetLoadingScheme2D,

}



# ==================== ERRORS ==================== #

# ---------- Basic Function Sets ---------- #
# NOTE: BasicFunctionSets are not bases as we do not require them to be 
# linearly independent. They are used to build more complex 
# SpaceTimeFunctionSets.

class BasicFunctionSet(ABC):

    def __init__(
            self,
            domain_range: DomainRange,
            n_fcns: int, 
            ndim: int = 1
        ) -> None:
        self.domain_range = domain_range
        self.n_fcns = n_fcns
        self.ndim = ndim

    @abstractmethod
    def __call__(self, eval_points: torch.Tensor) -> torch.Tensor:
        pass
    

class BSplineBasicFS(BasicFunctionSet):

    def __init__(
            self,
            domain_range: List[float],
            n_fcns: int,
            order: int
        ) -> None:
        super().__init__(domain_range=domain_range, n_fcns=n_fcns)
        self.fcns = bspline_basis_fcns(n_fcns, domain_range, order)

    def __call__(
            self,
            eval_points: torch.Tensor
        ) -> torch.Tensor:
        mask1 = eval_points >= self.domain_range[0]
        mask2 = eval_points <= self.domain_range[1]
        mask = torch.logical_and(mask1, mask2)
        out = torch.zeros((len(self.fcns), len(eval_points)), dtype=torch.float64)
        out_list = [None] * self.n_fcns
        for j in range(self.n_fcns):
            out_ = self.fcns[j](eval_points[mask].numpy())
            out_ = torch.tensor(out_).to(torch.float64)
            out_list[j] = out_
        out[:,mask] = torch.vstack(out_list)
        return out
    
    
class BSplinePinnedBasicFS(BSplineBasicFS):

        def __init__(
            self,
            domain_range: List[float],
            n_fcns: int,
            order: int
        ) -> None:
            super().__init__(domain_range=domain_range, n_fcns=n_fcns + 2, order=order)
            self.fcns = self.fcns[1:(n_fcns + 1)]  # exclude boundary functions
            self.n_fcns = n_fcns


# TODO: This doesn't play nicely with batching
class GaussianProcessBasicFS(BasicFunctionSet):

    def __init__(
            self, 
            domain_range: List[float],  # NOTE: Can I use domain_range to infer the number of time points? A bit hacky...
            n_fcns: int,
            kernel: Callable,
            gen: torch.Generator,
            chol_pen: float = 1e-5
        ) -> None:
        super().__init__(domain_range=domain_range, n_fcns=n_fcns)

        # NOTE: In order to enable batched evaluations of this function set, 
        # we simulate GPs on a sufficiently dense grid, then fit BSplines to
        # the simulated data. 

        # Simulate processes on a sufficiently dense grid
        grid = torch.linspace(
            domain_range[0], domain_range[1], 
            steps=domain_range[1],
            dtype=torch.float64
        )
        procs = simulate_gauss_procs(
            grid=grid,
            n_procs=n_fcns,
            kernel=kernel,
            gen=gen,
            gamma=chol_pen
        )

        # Get B-spline basis functions
        self.procs = [None] * n_fcns
        for j in range(n_fcns):
            knots, coeffs, deg = splrep(grid.numpy(), procs[j].numpy(), s=0, k=3)
            self.procs[j] = BSpline(knots, coeffs, deg, extrapolate=False)

    def __call__(self, eval_points: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(self.n_fcns, len(eval_points), dtype=torch.float64)
        for j in range(self.n_fcns):
            out[j] = torch.tensor(self.procs[j](eval_points)).to(torch.float64)
        return out


class Bump1DBasicFS(BasicFunctionSet):

    def __init__(
            self, 
            domain_range: List[float],
            n_fcns: int, 
            width: float
        ) -> None:
        super().__init__(domain_range=domain_range, n_fcns=n_fcns)
        centers = torch.linspace(0, 1, n_fcns, dtype=torch.float64)
        self.fcns = [BumpFunction1D(c.item(), width / 2, 1) for c in centers]

    def __call__(
            self, 
            eval_points: torch.Tensor
        ) -> torch.Tensor:
        out = torch.zeros((self.n_fcns, len(eval_points)), dtype=torch.float64)
        for j in range(self.n_fcns):
            out[j] = self.fcns[j](eval_points)
        return out


class BSplinePinned1DBasicFS(BasicFunctionSet):

    def __init__(
            self,
            domain_range: List[float],
            n_fcns: int,
            width: float,
            gen: torch.Generator  
        ) -> None:
        super().__init__(domain_range=domain_range, n_fcns=n_fcns)

        # Create n_fcns "local bases" from which this set's functions will 
        # be generated
        n_fcns_base = 5
        self.local_bases = [None] * n_fcns

        left_endpoints = torch.linspace(
            start=domain_range[0] - width/2, 
            end=domain_range[1] - width/2, 
            steps=n_fcns
        )
        for j in range(n_fcns):
            self.local_bases[j] = BSplinePinnedBasicFS(
                [left_endpoints[j], left_endpoints[j] + width], 
                n_fcns_base, 4
            )

        # Generate coefficients that will be used generate single functions 
        # from local bases
        self.coeffs = torch.randn(
            size=(n_fcns, n_fcns_base), 
            dtype=torch.float64, 
            generator=gen
        )

    def __call__(
            self,
            eval_points: torch.Tensor
        ) -> torch.Tensor:
        out = torch.zeros((self.n_fcns, len(eval_points)))
        for j in range(self.n_fcns):
            temp = self.local_bases[j](eval_points)
            out[j] = self.coeffs[j] @ temp
        return out
    

class TensorBasicFS(BasicFunctionSet):

    def __init__(self, fsets: List[BasicFunctionSet]) -> None:

        # Check that all function sets are 1-dimensional
        if not all([f.ndim == 1 for f in fsets]):
            raise Exception("All function sets must be 1-dimensional!")
        self.fsets = fsets
        
        # Call BasicFunctionSet.__init__
        domain_range = []
        n_fcns = 1
        for fset in fsets:
            domain_range.append(fset.domain_range)
            n_fcns *= fset.n_fcns
        super().__init__(domain_range=domain_range, n_fcns=n_fcns, ndim=len(fsets))  
    
    def __call__(self, eval_points: torch.Tensor) -> torch.Tensor:

        if eval_points.size(1) != self.ndim: 
            raise Exception("Number of columns in eval_points must equal n_fsets!")

        # Evaluate each fset at the appropriate column of eval_points
        evals = [
            self.fsets[d](eval_points[:,d]) for d in range(self.ndim)
        ]

        # Get tensor product
        out = []
        fset_fidx = [list(range(f.n_fcns)) for f in self.fsets]  # Ex: [[0, 1], [0, 1, 2]]
        for fidx in itertools.product(*fset_fidx):
            fset_outs_ = [
                evals[d][fidx[d]] for d in range(self.ndim)
            ]
            fset_outs_ = torch.vstack(fset_outs_)
            out.append(torch.prod(fset_outs_, dim=0))
        return torch.vstack(out)
    


# ---------- Space-Time Function Sets ---------- #
    
class SpaceTimeFunctionSet(ABC):

    def __init__(
            self, 
            width_space: float,
            n_time: int,
            kernel_length_time: float,
            gen: torch.Generator = None
        ) -> None:
        self.gen = gen
        self.width_space = width_space
        self.n_time = n_time
        self.kernel_length_time = kernel_length_time

        self.fset_space = self.get_space_fset()
        self.fset_time = self.get_time_fset()

        if self.fset_space.n_fcns != self.fset_time.n_fcns:
            raise Exception("Number of functions in spatial and temporal " 
                            "function sets must be equal!")
        
        self.n_fcns = self.fset_space.n_fcns
        self.ndim = self.fset_space.ndim + 1
        
    def __call__(self, eval_points: torch.Tensor) -> torch.Tensor: 

        if eval_points.size(1) != self.ndim:
            raise Exception("The number of columns in eval_points does not " 
                            "match the dimension of this SpaceTimeFunctionSet!")

        points_time = eval_points[:, 0]
        points_space = eval_points[:, 1:]

        # Remove empty dimension if points_space is m-by-1
        if points_space.size(1) == 1:
            points_space = points_space[:,0]

        return self.fset_time(points_time) * self.fset_space(points_space)

 
    @abstractmethod
    def get_space_fset(self) -> BasicFunctionSet:
        pass

    @abstractmethod
    def get_time_fset(self) -> BasicFunctionSet:
        pass


class GaussProc_Bump1D_SpaceTimeFS(SpaceTimeFunctionSet):

    def get_space_fset(self) -> BasicFunctionSet:
        return Bump1DBasicFS(
            domain_range=[0, 1], 
            n_fcns=20, 
            width=self.width_space
        )
    
    def get_time_fset(self) -> BasicFunctionSet:
        return GaussianProcessBasicFS(
            domain_range=[1, self.n_time],
            n_fcns=20,
            kernel=partial(squared_exponential_kernel, length=self.kernel_length_time),
            gen=gen
        )
    

class GaussProc_BumpTensor2D_SpaceTimeFS(SpaceTimeFunctionSet):

    def get_space_fset(self) -> BasicFunctionSet:
        fset = Bump1DBasicFS(
            domain_range=[0, 1], 
            n_fcns=20, 
            width=self.width_space
        )
        return TensorBasicFS([fset, fset])

    def get_time_fset(self) -> BasicFunctionSet:
        return GaussianProcessBasicFS(
            domain_range=[1, self.n_time],
            n_fcns=400,
            kernel=partial(squared_exponential_kernel, length=self.kernel_length_time),
            gen=gen
        )
    

class GaussProc_BSplinePinned1D_SpaceTimeFS(SpaceTimeFunctionSet):

    def get_space_fset(self) -> BasicFunctionSet:
        return BSplinePinned1DBasicFS(
            domain_range=[0, 1],
            n_fcns=20,
            width=self.width_space,
            gen=self.gen
        )

    def get_time_fset(self) -> BasicFunctionSet:
        return GaussianProcessBasicFS(
            domain_range=[1, self.n_time],
            n_fcns=20,
            kernel=partial(squared_exponential_kernel, length=self.kernel_length_time),
            gen=gen
        )
    

class GaussProc_BSplinePinnedTensor2D_SpaceTimeFS(SpaceTimeFunctionSet):

    def get_space_fset(self) -> BasicFunctionSet:
        fset = BSplinePinned1DBasicFS(
            domain_range=[0, 1],
            n_fcns=20,
            width=self.width_space,
            gen=self.gen
        )
        return TensorBasicFS([fset, fset])

    def get_time_fset(self) -> BasicFunctionSet:
        return GaussianProcessBasicFS(
            domain_range=[1, self.n_time],
            n_fcns=400,
            kernel=partial(squared_exponential_kernel, length=self.kernel_length_time),
            gen=gen
        )



# ---------- Error Schemes ---------- #
# NOTE: Error schemes are defined by a `fset`, and a range 
# (`scale_min`, `scale_mix`) from which to draw `n_fcns` scaling factors. 

class ErrorScheme(ABC):
        
    def __init__(
            self, 
            width_space: float,
            n_time: int,
            gen: torch.Generator
        ) -> None:

        # Construct function set
        self.fset = self.get_fset(
            width_space=width_space, 
            n_time=n_time,
            gen=gen
        )
        self.n_fcns = self.fset.n_fcns
        self.ndim = self.fset.ndim

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
            vals = self.fset(points_)  # n_fcns-by-sz

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
    def get_fset(
            self, 
            width_space: float, 
            n_time: int, 
            kernel_length_time: float
        ) -> SpaceTimeFunctionSet:
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
    
class GaussProc_Bump1D_ErrorScheme(ErrorScheme):

    scale_min = 0.1
    scale_max = 1

    def get_fset(self, width_space, n_time, gen):
        return GaussProc_Bump1D_SpaceTimeFS(
            width_space=width_space,
            n_time=n_time,
            kernel_length_time=5,
            gen=gen
        )
    

class GaussProc_BumpTensor2D_ErrorScheme(ErrorScheme):

    scale_min = 0.1
    scale_max = 1

    def get_fset(self, width_space, n_time, gen):
        return GaussProc_BumpTensor2D_SpaceTimeFS(
            width_space=width_space,
            n_time=n_time,
            kernel_length_time=5,
            gen=gen
        )
    

class GaussProc_BSplinePinned1D_ErrorScheme(ErrorScheme):

    scale_min = 0.1
    scale_max = 1

    def get_fset(self, width_space, n_time, gen):
        return GaussProc_BSplinePinned1D_SpaceTimeFS(
            width_space=width_space,
            n_time=n_time,
            kernel_length_time=5,
            gen=gen
        )
    

class GaussProc_BSplinePinnedTensor2D_ErrorScheme(ErrorScheme):

    scale_min = 0.1
    scale_max = 1

    def get_fset(self, width_space, n_time, gen):
        return GaussProc_BSplinePinnedTensor2D_SpaceTimeFS(
            width_space=width_space,
            n_time=n_time,
            kernel_length_time=5,
            gen=gen
        )
    

ERROR_SCHEMES = {

    # 1-dimensional
    'GaussProc_Bump1D': GaussProc_Bump1D_ErrorScheme,
    'GaussProc_BSplinePinned1D': GaussProc_BSplinePinned1D_ErrorScheme,

    # 1-dimensional
    'GaussProc_BumpTensor2D': GaussProc_BumpTensor2D_ErrorScheme,
    'GaussProc_BSplinePinnedTensor2D': GaussProc_BSplinePinnedTensor2D_ErrorScheme,

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
    """Returns a (sparse) error tensor of dimension n_err_fcns-by-n_time-bysz_space."""
    
    # NOTE: `vals` is a sparse tensor of dimension n_err_fcns-by-n, where n is the 
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
    # NOTE: By scaling each error function by a coefficient drawn from a 
    # zero-mean distribution with unit variance, we permit a simple closed 
    # form for the spatial error covariance which we use later. 
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

    # Yield the global and local norms for scaling of "truth"
    yield norm_global, norm_local

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
        '--dir_dataset',
        help="Dataset directory in which simulated data will be stored."
    )
    parser.add_argument(
        '--dir_out',
        help=("Output directory in which loadings, factors, and error "
              "covariance will be stored.")
    )
    parser.add_argument(
        '--n_time', type=int,
        help=("Number of time points for which to simulate data. Since the "
              "temporal domain is 1-dimensional, this an integer argument.")
    )
    parser.add_argument(
        '--sz_space', nargs='+', type=int,
        help="Shape of the spatial grid on which to simulate data."
    )
    parser.add_argument(
        '--factor_kernel_length', type=int,
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
        '--n_facs', type=int,
        help="Number of factors in global component."
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
        help="Maximumum number of time points to include in each file."
    )
    parser.add_argument(
        '--refresh_dirs', action='store_true',
        help="Flag for refreshing /datasets and /out directories."
    )
    parser.add_argument(
        '--dir_truth', type=str,
        help=("Directory in which to store true loadings, factors, and "
              "errors.If None, then do not save.")
    )
    parser.add_argument(
        '--seed', default=12345, type=int,
        help="Integer used to seed generator."
    )
    args = parser.parse_args()

    # Configure globals
    config = load_config(args.config)
    gen = torch.Generator().manual_seed(args.seed)
    load_scheme = LOADING_SCHEMES[args.load_scheme](args.n_facs)
    err_scheme = ERROR_SCHEMES[args.err_scheme](args.delta, args.n_time, gen)
    sz = [args.n_time] + args.sz_space

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

    # Refresh directories
    if args.refresh_dirs:
        refresh_directory(args.dir_dataset)
        refresh_directory(args.dir_out)

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

    # Generate `points_time` and `indices_time` from `n_time`
    indices_time = torch.arange(args.n_time, dtype=torch.int32)
    points_time = (indices_time + 1).to(torch.float64)

    # Get `points` and `indices` by taking cartesian product
    indices = torch.cartesian_prod(indices_time, *indices_space_list)
    points = torch.cartesian_prod(points_time, *points_space_list)

    # Build loading tensor
    print("Preparing loadings...")
    vals = load_scheme(points_space)  # n_facs-by-sz_shape
    loads = build_loadings(indices_space, args.sz_space, vals)

    # Build factor tensor
    print("Preparing factors...")
    kernel = partial(squared_exponential_kernel, length=args.factor_kernel_length)
    facs = simulate_gauss_procs(points_time, args.n_facs, kernel, gen).t()  # n_time-by-n_facs

    # Build error tensor, then scale error tensor by coefficients
    # TODO: Is there a way to accelerate error generation? Currently much slower
    # than loading/factor generation. Cache error schemes?
    print("Preparing errors....")
    vals = err_scheme(points)  
    errs = build_errors(indices, sz, vals)  # n_fcns-by-n_time-by-sz_shape (sparse)


    # ---------- DATA SIMULATION ---------- #

    print("Simulating new data...")
    dataloader = simulate_ffm_data(
        loads, 
        facs,
        errs,
        args.prop_global,
        args.batch_size,
        gen=gen
    )
    norm_global, norm_local = next(dataloader)
    write_generated_tensor(dataloader, args.dir_dataset, 'data-full')

    # Write (properly scaled) tensors
    # NOTE: (Loading and error scaling)
    #   To control the signal-to-noise ratio, we scale the global and local
    #   components of our data by p/||comp_glob|| and (1-p)/||comp_loc||, 
    #   respectively. This means that the "true" loadings and errors are not
    #   `loads` and `errs`, but these quantities scaled by the aforementioned
    #   factors. 
    if args.dir_truth is not None:
        n_vars = multiply_list(loads.shape[1:])
        loads = loads.reshape(loads.shape[0], n_vars)
        errs = reshape_sparse_coo_tensor(errs, [*errs.shape[:2], n_vars])
        torch.save(
            loads * args.prop_global / norm_global, 
            os.path.join(args.dir_truth, 'loads.pt')
        )
        torch.save(facs, os.path.join(args.dir_truth, 'facs.pt'))
        torch.save(
            errs * (1 - args.prop_global) / norm_local, 
            os.path.join(args.dir_truth, 'errs.pt')
        )

    print("DONE!")

    # NOTE: It takes 90 seconds to generate data for a 30-by-30 spatial grid on
    # 500 time points! Consider caching error schemes. 
