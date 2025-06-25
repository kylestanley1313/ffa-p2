import torch
import torch.nn as nn
from typing import Optional


__all__ = [
    'LowRankCovariance'
]


class LowRankCovariance(nn.Module):

    def __init__(
            self, 
            n_vars: int, 
            n_facs: int, 
            path_init: Optional[str] = None
        ):
        super().__init__()
        self.n_vars = n_vars
        self.n_facs = n_facs
        if path_init:
            self.loads = torch.load(path_init)
        else:
            self.loads = torch.randn(n_vars, n_facs, dtype=torch.float32)
        self.loads.requires_grad_()
        self.loads = nn.Parameter(self.loads)

    def forward(self, points):
        return (self.loads[points[:,0]] * self.loads[points[:,1]]).sum(dim=1)
    
    def get_loads(self, idx=None):
        if idx is None:
            return self.loads.data
        else:
            return self.loads.data[idx]
   
    def set_loads(self, loads, idx=None):
        if idx is None:
            self.loads.data = loads
        else:
            self.loads.data[idx] = loads