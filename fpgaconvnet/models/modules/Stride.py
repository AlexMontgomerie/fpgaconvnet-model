"""
"""

import math
import os
import sys
from typing import Union, List
from dataclasses import dataclass, field

import numpy as np
import pydot

from fpgaconvnet.models.modules import Module, MODULE_FONTSIZE

@dataclass
class Stride(Module):
    kernel_size: Union[List[int],int]
    stride: Union[List[int],int]
    backend: str = "chisel"
    regression_model: str = "linear_regression"

    def __post_init__(self):

        # format kernel size as a 2 element list
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size, self.kernel_size]
        elif isinstance(self.kernel_size, list):
            assert len(self.kernel_size) == 2, "Must specify two kernel dimensions"
        else:
            raise TypeError

        # format stride as a 2 element list
        if isinstance(self.stride, int):
            self.stride = [self.stride, self.stride]
        elif isinstance(self.stride, list):
            assert len(self.stride) == 2, "Must specify two stride dimensions"
        else:
            raise TypeError

        # load the resource model coefficients
        self.rsc_coef["LUT"] = np.load(
                os.path.join(os.path.dirname(__file__),
                f"../../coefficients/{self.regression_model}/{self.backend}/fork_lut.npy"))
        self.rsc_coef["FF"] = np.load(
                os.path.join(os.path.dirname(__file__),
                f"../../coefficients/{self.regression_model}/{self.backend}/fork_ff.npy"))
        self.rsc_coef["BRAM"] = np.load(
                os.path.join(os.path.dirname(__file__),
                f"../../coefficients/{self.regression_model}/{self.backend}/fork_bram.npy"))
        self.rsc_coef["DSP"] = np.load(
                os.path.join(os.path.dirname(__file__),
                f"../../coefficients/{self.regression_model}/{self.backend}/fork_dsp.npy"))

    def utilisation_model(self):
        return {
            "LUT"  : np.array([math.ceil(math.log(self.channels*self.rows*self.cols,2))]),
            "FF"   : np.array([math.ceil(math.log(self.channels*self.rows*self.cols,2))]),
            "DSP"  : np.array([1]),
            "BRAM" : np.array([1]),
        }

    def module_info(self):
        # get the base module fields
        info = Module.module_info(self)
        # add module-specific info fields
        info["coarse"] = self.coarse
        info["kernel_size"] = self.kernel_size
        info["stride"] = self.stride
        # return the info
        return info

    def visualise(self, name):
        return pydot.Node(name,label="fork", shape="box",
                style="filled", fillcolor="azure",
                fontsize=MODULE_FONTSIZE)

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"
        assert data.shape[3] == self.kernel_size[0]  , "ERROR: invalid column dimension"
        assert data.shape[4] == self.kernel_size[1]  , "ERROR: invalid column dimension"

        out = np.ndarray((
            self.rows//self.stride[0],
            self.cols//self.stride[1],
            self.channels,
            self.kernel_size[0],
            self.kernel_size[1]),dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = data[
              index[0]*self.stride[0],
              index[1]*self.stride[1],
              index[2],index[3],index[4]]

        return out

