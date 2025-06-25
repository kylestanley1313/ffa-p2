import os
import torch

from utils import read_tensors


__all__ = [
    'CovarianceComputer'
]


class CovarianceComputer(object):

    def __init__(self, rank: int, dir_cov: str, dir_data: str) -> None:
        self.rank = rank
        self.path_points = os.path.join(dir_cov, f'points-{rank}.pt')
        self.path_cov = os.path.join(dir_cov, f'cov-{rank}.pt')
        self.dir_data = dir_data

    def compute(self):
        points = torch.load(self.path_points)
        num_points = len(points)
        t1 = torch.zeros(num_points, dtype=torch.float64)
        t2 = torch.zeros(num_points, dtype=torch.float64)
        t3 = torch.zeros(num_points, dtype=torch.float64)
        n = 0
        dataloader = read_tensors(self.dir_data, 'data')
        for batch in dataloader: 
            n += len(batch)
            for i in range(num_points):
                row, col = points[i, :]
                t1[i] += torch.sum(batch[:,row] * batch[:,col])
                t2[i] += torch.sum(batch[:,row])
                t3[i] += torch.sum(batch[:,col])
        cov = (t1 - t2 * t3 / n) / (n - 1)
        torch.save(cov, self.path_cov)

