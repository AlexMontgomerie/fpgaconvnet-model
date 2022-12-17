import os
from typing import Union, List
import numpy as np
from fpgaconvnet.models.modules import Pool

class MaxPool(Pool):
    def __init__(self, rows: int, cols: int, channels: int,
            kernel_size: Union[List[int],int], backend: str = "chisel"):
        super().__init__(rows, cols, channels, kernel_size,
                pool_type="max", backend=backend)

        # get the cache path
        rsc_cache_path = os.path.dirname(__file__) + \
                f"/../../coefficients/{self.regression_model}/{self.backend}"

        # iterate over resource types
        self.rsc_coef = {}
        for rsc_type in self.utilisation_model():
            coef_path = os.path.join(rsc_cache_path, f"maxpool_{rsc_type}.npy".lower())
            self.rsc_coef[rsc_type] = np.load(coef_path)

    def utilisation_model(self):
        if self.backend == "hls":
            pass
        elif self.backend == "chisel":
            return {
                "Logic_LUT"  : np.array([
                    self.kernel_size[0],self.kernel_size[1],
                ]),
                "LUT_RAM"  : np.array([1]),
                "LUT_SR"  : np.array([1]),
                "FF"   : np.array([
                    self.kernel_size[0],self.kernel_size[1],
                ]),
                "DSP"  : np.array([1]),
                "BRAM36" : np.array([1]),
                "BRAM18" : np.array([1]),
            }
        else:
            raise ValueError()


