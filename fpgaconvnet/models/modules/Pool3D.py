"""
This module performs the max pooling function
across a kernel-size window of the feature map.

.. figure:: ../../../figures/pool_max_diagram.png
"""
import math
import os
import sys
from typing import Union, List
from dataclasses import dataclass, field

import numpy as np
import pydot

from fpgaconvnet.models.modules import Module3D, MODULE_3D_FONTSIZE

@dataclass
class Pool3D(Module3D):
    # kernel_size: Union[List[int], int]
    kernel_rows: int
    kernel_cols: int
    kernel_depth: int
    pool_type: str = "max"
    backend: str = "chisel"

    def __post_init__(self):
        self.__class__.__name__ = f"{self.pool_type.capitalize()}Pool3D"
        # get the cache path
        rsc_cache_path = os.path.dirname(__file__) + \
                f"/../../coefficients/{self.backend}"

        # iterate over resource types
        self.rsc_coef = {}
        for rsc_type in self.utilisation_model():
            # load the resource coefficients from the 2D version
            coef_path = os.path.join(rsc_cache_path, f"{self.__class__.__name__.split('3D')[0]}_{rsc_type}.npy".lower())
            self.rsc_coef[rsc_type] = np.load(coef_path)

    def utilisation_model(self):
        if self.pool_type == 'max':
            if self.backend == "hls":
                pass
            elif self.backend == "chisel":
                # TODO: Update this to use the 3D version FIXME
                # Logic_LUT and FF should be arrays of size 3 not 2
                return {
                    "Logic_LUT"  : np.array([
                        self.kernel_rows,
                        self.kernel_cols*self.kernel_depth
                    ]),
                    "LUT_RAM"  : np.array([1]),
                    "LUT_SR"  : np.array([1]),
                    "FF"   : np.array([
                        self.kernel_rows,
                        self.kernel_cols*self.kernel_depth
                    ]),
                    "DSP"  : np.array([1]),
                    "BRAM36" : np.array([1]),
                    "BRAM18" : np.array([1]),
                }
            else:
                raise ValueError()
        else:
            raise NotImplementedError()

    def module_info(self):
        # get the base module fields
        info = Module3D.module_info(self)
        # add module-specific info fields
        # info["kernel_size"] = self.kernel_size
        info["kernel_rows"] = self.kernel_rows
        info["kernel_cols"] = self.kernel_cols
        info["kernel_depth"] = self.kernel_depth
        info["pool_type"] = 0 if self.pool_type == 'max' else 1
        # return the info
        return info

    def visualise(self, name):
        return pydot.Node(name,label=f"{self.pool_type}pool3d",
                shape="box",
                height=self.kernel_rows,
                width=self.kernel_cols,
                depth=self.kernel_depth,
                style="filled", fillcolor="cyan",
                fontsize=MODULE_3D_FONTSIZE)

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth    , "ERROR: invalid depth dimension"
        assert data.shape[3] == self.channels, "ERROR: invalid channel dimension"
        assert data.shape[4] == self.kernel_rows  , "ERROR: invalid kernel size (x) dimension"
        assert data.shape[5] == self.kernel_cols  , "ERROR: invalid kernel size (y) dimension"
        assert data.shape[6] == self.kernel_depth  , "ERROR: invalid kernel size (z) dimension"

        out = np.ndarray((
            self.rows,
            self.cols,
            self.depth,
            self.channels),dtype=float)

        for index,_ in np.ndenumerate(out):
            if self.pool_type == 'max':
                out[index] = np.max(data[index])
            elif self.pool_type == 'avg':
                out[index] = np.mean(data[index])

        return out

