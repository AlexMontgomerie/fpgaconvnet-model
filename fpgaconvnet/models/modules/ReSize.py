"""
"""

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Union, List
import importlib

import numpy as np
import pydot

from fpgaconvnet.models.modules import int2bits, Module, MODULE_FONTSIZE

@dataclass
class ReSize(Module):
    scales: List[int]
    backend: str = "chisel"
    regression_model: str = "linear_regression"
    streams: int = 1
    latency_mode: int = False

    def __post_init__(self):
        pass

    def rows_out(self):
        return int(self.rows * self.scales[0])

    def cols_out(self):
        return int(self.cols * self.scales[1])

    def channels_out(self):
        return int(self.channels * self.scales[2])

    def rate_in(self):
        return np.prod(self.scales)

    def module_info(self):
        # get the base module fields
        info = Module.module_info(self)
        # add module-specific info fields
        info['scales'] = self.scales
        # return the info
        return info

    def pipeline_depth(self):
        return self.cols * self.channels * (self.scales[0]-1) + self.channels * (self.scales[1]-1)

    def memory_usage(self):
        return int(self.filters/self.groups)*self.data_width

    def utilisation_model(self):
        return {
            "Logic_LUT" : np.array([1]),
            "LUT_RAM"   : np.array([1]),
            "LUT_SR"    : np.array([0]),
            "FF"        : np.array([1]),
            "DSP"       : np.array([0]),
            "BRAM36"    : np.array([0]),
            "BRAM18"    : np.array([0]),
        }

    def functional_model(self, data):

        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"

        out = np.zeros((
            self.rows*self.scales[0],
            self.cols*self.scales[1],
            self.channels*self.scales[2]),dtype=float)

        for index, _ in np.ndenumerate(out):
                out[index] = data[
                        index[0]//self.scales[0],
                        index[1]//self.scales[1],
                        index[2]//self.scales[2],
                    ]

        return out

