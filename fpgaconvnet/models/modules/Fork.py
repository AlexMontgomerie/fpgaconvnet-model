"""
The Fork module provides functionality for
parallelism within layers. By duplicating the
streams, it can be used for exploiting
parallelism across filters in the Convolution
layers.

.. figure:: ../../../figures/fork_diagram.png
"""

import numpy as np
import math
import os
import sys
from typing import Union, List
from dataclasses import dataclass, field

from fpgaconvnet.models.modules import Module

@dataclass
class Fork(Module):
    kernel_size: Union[List[int],int]
    coarse: int

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
                "../../coefficients/fork_lut.npy"))
        self.rsc_coef["FF"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/fork_ff.npy"))
        self.rsc_coef["BRAM"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/fork_bram.npy"))
        self.rsc_coef["DSP"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/fork_dsp.npy"))

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
        # return the info
        return info

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"
        assert data.shape[3] == self.kernel_size[0]  , "ERROR: invalid column dimension"
        assert data.shape[4] == self.kernel_size[1]  , "ERROR: invalid column dimension"

        out = np.ndarray((
            self.rows,
            self.cols,
            self.channels,
            self.coarse,
            self.kernel_size[0],
            self.kernel_size[1]),dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = data[
              index[0],
              index[1],
              index[2],
              index[4],
              index[5]]

        return out

