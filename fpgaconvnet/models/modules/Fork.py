"""
The Fork module provides functionality for
parallelism within layers. By duplicating the
streams, it can be used for exploiting
parallelism across filters in the Convolution
layers.

.. figure:: ../../../figures/fork_diagram.png
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
class Fork(Module):
    kernel_size: Union[List[int],int]
    coarse: int
    backend: str = "chisel"
    regression_model: str = "linear_regression"
    streams: int = 1

    def __post_init__(self):

        # format kernel size as a 2 element list
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size, self.kernel_size]
        elif isinstance(self.kernel_size, list):
            assert len(self.kernel_size) == 2, "Must specify two kernel dimensions"
        else:
            raise TypeError

        # perform basic module initialisation
        Module.__post_init__(self)

    def module_info(self):
        # get the base module fields
        info = Module.module_info(self)
        # add module-specific info fields
        info["coarse"] = self.coarse
        info["kernel_size"] = self.kernel_size
        # return the info
        return info

    def utilisation_model(self):
        if self.backend == "hls":
            pass # TODO
        elif self.backend == "chisel":
            return {
                "Logic_LUT" : np.array([
                    self.streams*np.prod(self.kernel_size), # input buffer
                    self.streams*self.data_width*np.prod(self.kernel_size), # input buffer
                    self.streams*np.prod(self.kernel_size)*self.coarse, # output buffer
                    self.streams*self.data_width*np.prod(self.kernel_size)*self.coarse, # output buffer
                    1,
                ]),
                "LUT_RAM"   : np.array([0]),
                "LUT_SR"    : np.array([0]),
                "FF"    : np.array([
                    self.streams*np.prod(self.kernel_size), # input buffer (ready)
                    self.streams*self.data_width*np.prod(self.kernel_size)*self.coarse, # output buffer (data)
                    1,
                ]),
                "DSP"       : np.array([0]),
                "BRAM36"    : np.array([0]),
                "BRAM18"    : np.array([0]),
            }
        else:
            raise ValueError(f"{self.backend} backend not supported")

    def get_pred_array(self):
        return np.array([
        self.data_width, self.data_width//2,
        self.coarse, np.prod(self.kernel_size), self.streams
        ]).reshape(1,-1)

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

