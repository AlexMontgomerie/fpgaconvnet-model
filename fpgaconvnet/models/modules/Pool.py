"""
This module performs the max pooling function
across a kernel-size window of the feature map.

.. figure:: ../../../figures/pool_max_diagram.png
"""
import numpy as np
import math
import os
import sys
from typing import Union, List
from dataclasses import dataclass, field

from fpgaconvnet.models.modules import Module

@dataclass
class Pool(Module):
    kernel_size: Union[List[int],int]
    pool_type: str = "max"

    def __post_init__(self):
        # format kernel size as a 2 element list
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size, self.kernel_size]
        elif isinstance(self.kernel_size, list):
            assert len(self.kernel_size) == 2, "Must specify two kernel dimensions"
        else:
            raise TypeError

        # load the resource model coefficients
        self.rsc_coef["LUT"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/pool_lut.npy"))
        self.rsc_coef["FF"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/pool_ff.npy"))
        self.rsc_coef["BRAM"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/pool_bram.npy"))
        self.rsc_coef["DSP"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/pool_dsp.npy"))

    def utilisation_model(self):
        return {
            "LUT"  : np.array([self.kernel_size[0],self.kernel_size[1],self.cols,self.rows,self.channels,self.data_width]),
            "FF"   : np.array([self.kernel_size[0],self.kernel_size[1],self.cols,self.rows,self.channels,self.data_width]),
            "DSP"  : np.array([1]),
            "BRAM" : np.array([1]),
        }

    def module_info(self):
        # get the base module fields
        info = Module.module_info(self)
        # add module-specific info fields
        info["kernel_size"] = self.kernel_size
        info["pool_type"] = 0 if self.pool_type == 'max' else 1
        # return the info
        return info

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"
        assert data.shape[3] == self.kernel_size[0]  , "ERROR: invalid kernel size (x) dimension"
        assert data.shape[4] == self.kernel_size[1]  , "ERROR: invalid kernel size (y) dimension"

        out = np.ndarray((
            self.rows,
            self.cols,
            self.channels),dtype=float)

        for index,_ in np.ndenumerate(out):
            if self.pool_type == 'max':
                out[index] = np.max(data[index])
            elif self.pool_type == 'avg':
                out[index] = np.mean(data[index])

        return out

