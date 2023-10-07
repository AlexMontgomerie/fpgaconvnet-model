"""
The bias module implements the addition of the
bias term for the convolution (and inner product)
layers when applicable. Each of element of the bias
vector is added to its corresponding output feature
map.

Figure pending.
"""

import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pydot

from fpgaconvnet.models.modules import int2bits, Module, MODULE_FONTSIZE

@dataclass
class Bias(Module):
    filters: int
    biases_width: int = field(default=16, init=False)
    backend: str = "chisel"
    regression_model: str = "linear_regression"
    streams: int = 1

    def channels_in(self):
        return self.filters

    def channels_out(self):
        return self.filters

    def module_info(self):
        # get the base module fields
        info = Module.module_info(self)
        # add module-specific info fields
        info['filters'] = self.filters
        # return the info
        return info

    def utilisation_model(self):

        if self.backend == "hls":
            return {
                "LUT"   : np.array([1]),
                "FF"    : np.array([1]),
                "DSP"   : np.array([0]),
                "BRAM"  : np.array([0]),
            }

        if self.backend == "chisel":
            return {
                "Logic_LUT" : np.array([
                        self.streams*self.data_width,
                    ]),
                "LUT_RAM"   : np.array([
                        self.streams*self.data_width*self.channels,
                    ]),
                "LUT_SR"    : np.array([0]),
                "FF"        : np.array([
                        self.streams*self.data_width,
                        int2bits(self.channels),
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
        self.filters, self.channels
        ]).reshape(1,-1)

    def visualise(self, name):
        return pydot.Node(name,label="bias", shape="box",
                style="filled", fillcolor="chartreuse",
                fontsize=MODULE_FONTSIZE)

    def functional_model(self, data, biases):
        # check input dimensionality
        assert data.shape[0] == self.rows                   , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols                   , "ERROR: invalid column dimension"
        assert data.shape[2] == self.filters                , "ERROR: invalid filter dimension"
        # check bias dimensionality
        assert biases.shape[0] == self.filters              , "ERROR: invalid filter dimension"

        out = np.zeros((
            self.rows,
            self.cols,
            self.filters,
            ), dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = data[index] + biases[index[2]]

        return out
