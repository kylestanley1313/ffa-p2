import argparse
import time
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.profiler import profiler, record_function
from typing import Optional

from utils import gen_points, multiply_list


def objective(
        preds: torch.Tensor, 
        cov: torch.Tensor
    ):
    return mse_loss(preds, cov)


class LowRankCovariance(nn.Module):

    def __init__(
            self, 
            num_vars: int, 
            num_facs: int, 
            path_init: Optional[str] = None
        ):
        super().__init__()
        self.num_facs = num_facs
        self.loads = nn.Embedding(num_vars, num_facs, dtype=torch.float64)
        if path_init:
            self.loads.weight.data = torch.load(path_init)

    def forward(self, idx0, idx1):
        loads0 = self.loads(idx0)
        loads1 = self.loads(idx1)
        return (loads0 * loads1).sum(dim=1)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_shape', type=int, nargs='+')
    args = parser.parse_args()

# Globals
num_facs = 4
num_vars = multiply_list(args.grid_shape)
delta = 0.1
batch_size = int(1.2 * num_vars)
lr = 0.1

# Core objects
points_loader = gen_points(args.grid_shape, delta, batch_size)
model = LowRankCovariance(num_vars, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

iter = 0
time_backward = 0
for points in points_loader: 

    # Dummy covariance
    cov = torch.randn(len(points), dtype=torch.float64)

    # Forward pass
    preds = model(points[:,0], points[:,1])
    loss = objective(preds, cov)

    # Backward pass
    optimizer.zero_grad()
    start = time.time()
    loss.backward()
    end = time.time()
    time_backward += end - start
    iter += 1

print(f"Avg. backward pass time: {time_backward / iter}")
