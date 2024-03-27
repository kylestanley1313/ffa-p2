import argparse
import numpy as np
import os
import torch
from typing import Optional

from factor_analyzer.rotator import Rotator


class PostProcesser(object):

    def __init__(
            self,
            rotate: bool = True,
            shrink: bool = True,
            max_iter: int = 1000,
            tol: float = 1e-5
        ) -> None:
        self.rotate = rotate
        self.shrink = shrink
        if self.rotate: 
            self.rotator = Rotator(method='varimax', max_iter=max_iter, tol=tol)

    def __call__(
            self, 
            loads: torch.Tensor, 
            kappa: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        out = loads.numpy()
        if self.rotate: 
            out = self._rotate(out)
        if self.shrink:
            out = self._shrink(out, kappa.numpy())
        return torch.from_numpy(out)

    def _rotate(self, loads: np.ndarray) -> np.ndarray:
        return self.rotator.fit_transform(loads)

    def _shrink(self, loads: np.ndarray, kappa: np.ndarray) -> np.ndarray:

        num_facs = loads.shape[1]
        if len(kappa) != num_facs:
            raise Exception("Length of `kappa` must equal the number of factors in `loads`!")

        for k in range(num_facs):
            temp = np.abs(loads[k]) - kappa[k]/np.abs(loads[k])**2
            temp[temp < 0] = 0
            loads[k] = np.sign(loads[k]) * temp
        
        return loads
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_out', type=str)
    args = parser.parse_args()

    # Set paths
    dir_out = os.path.join('out', args.dir_out)
    path_in = os.path.join(dir_out, 'cov-model.pth')
    path_out = os.path.join(dir_out, 'cov-model-pp.pth')
    kappa = torch.tensor([0, 0], dtype=torch.float64)  # TODO: Temporary global

    # TODO: Should post-processing reside within the model itself? 

    # Post-process loadings
    pp = PostProcesser()
    model = torch.load(path_in)
    loads_pp = pp(model['loads'], kappa)
    model['loads'] = loads_pp
    torch.save(model, path_out)

