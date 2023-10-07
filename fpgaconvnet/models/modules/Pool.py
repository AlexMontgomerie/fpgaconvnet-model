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

from fpgaconvnet.models.modules import int2bits, Module, MODULE_FONTSIZE
from fpgaconvnet.tools.resource_analytical_model import queue_lutram_resource_model

@dataclass
class Pool(Module):
    kernel_size: Union[List[int],int]
    pool_type: str = "max"
    backend: str = "chisel"
    regression_model: str = "linear_regression"
    streams: int = 1

    def __name__(self):
        return f"{self.pool_type.capitalize()}Pool"

    def __post_init__(self):
        # format kernel size as a 2 element list
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size, self.kernel_size]
        elif isinstance(self.kernel_size, list):
            assert len(self.kernel_size) == 2, "Must specify two kernel dimensions"
        else:
            raise TypeError

        # get the module identifer
        self.module_identifier = self.__class__.__name__

        # load resource coefficients
        self.load_resource_coefficients(self.module_identifier)

    def utilisation_model(self):
        if self.backend == "hls":
            pass
        elif self.backend == "chisel":
            return {
                "Logic_LUT"  : np.array([
                    self.kernel_size[0]*self.kernel_size[1],
                    self.streams*self.data_width*self.kernel_size[0]*self.kernel_size[1], # tree buffer
                    self.streams*self.data_width*int2bits(self.kernel_size[0]*self.kernel_size[1]), # tree buffer
                    self.kernel_size[0],self.kernel_size[1], # input ready
                    1,
                ]),
                "LUT_RAM"  : np.array([
                    queue_lutram_resource_model(
                        int2bits(self.kernel_size[0]*self.kernel_size[1])+1, self.streams*self.data_width), # buffer
                    1,
                ]),
                "LUT_SR"  : np.array([0]),
                "FF"   : np.array([
                    self.streams*self.data_width, # output buffer
                    self.streams*self.data_width*self.kernel_size[0]*self.kernel_size[1], # op tree input
                    int2bits(self.kernel_size[0]*self.kernel_size[1]), # shift register
                    1,
                ]),
                "DSP"  : np.array([0]),
                "BRAM36" : np.array([0]),
                "BRAM18" : np.array([0]),
            }
        else:
            raise ValueError()

    def get_pred_array(self):
        return np.array([
        self.data_width, self.data_width//2,
        *self.kernel_size
        # self.pool_type,
        ]).reshape(1,-1)

    def module_info(self):
        # get the base module fields
        info = Module.module_info(self)
        # add module-specific info fields
        info["kernel_size"] = self.kernel_size
        info["pool_type"] = 0 if self.pool_type == 'max' else 1
        # return the info
        return info

    def visualise(self, name):
        return pydot.Node(name,label="pool", shape="box",
                height=self.kernel_size[0],
                width=self.kernel_size[1],
                style="filled", fillcolor="cyan",
                fontsize=MODULE_FONTSIZE)

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

