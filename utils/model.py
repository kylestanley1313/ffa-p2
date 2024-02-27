import torch
import torch.nn as nn
from typing import Optional


__all__ = [
    'LowRankCovariance'
]


class LowRankCovariance(nn.Module):

    def __init__(
            self, 
            num_vars: int, 
            num_facs: int, 
            path_init: Optional[str] = None
        ):
        super().__init__()
        self.num_facs = num_facs
        if path_init:
            self.loads = torch.load(path_init)
        else:
            self.loads = torch.randn(num_vars, num_facs, dtype=torch.float64)
        self.loads.requires_grad_()
        self.loads = nn.Parameter(self.loads)

    def forward(self, idx0, idx1):
        # loads0 = self.loads[idx0]
        # loads1 = self.loads[idx1]
        return (self.loads[idx0] * self.loads[idx1]).sum(dim=1)
    
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