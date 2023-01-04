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
from collections import namedtuple

import numpy as np
import pydot

from fpgaconvnet.models.modules import Module3D, MODULE_3D_FONTSIZE

from fpgaconvnet.models.modules import Fork

@dataclass
class Fork3D(Module3D):
    # kernel_size: Union[List[int], int]
    kernel_rows: int
    kernel_cols: int
    kernel_depth: int
    coarse: int
    backend: str = "chisel"
    regression_model: str = "linear_regression"
    streams: int = 1
    latency_mode: int = False
    block: int = False

    def __post_init__(self):

        # get the module identifer
        self.module_identifier = "Fork"

        # load resource coefficients
        self.load_resource_coefficients(self.module_identifier)

    @property
    def kernel_size(self):
        return [ self.kernel_rows, self.kernel_cols, self.kernel_depth ]

    @kernel_size.setter
    def kernel_size(self, kernel_size):
        self.kernel_rows = kernel_size[0]
        self.kernel_cols = kernel_size[1]
        self.kernel_depth = kernel_size[2]

    def module_info(self):
        # get the base module fields
        info = Module3D.module_info(self)
        # add module-specific info fields
        info["coarse"] = self.coarse
        # info["kernel_size"] = self.kernel_size
        info["kernel_rows"] = self.kernel_rows
        info["kernel_cols"] = self.kernel_cols
        info["kernel_depth"] = self.kernel_depth
        # return the info
        return info

    def utilisation_model(self):

        # load utilisation model from the 2D model
        self.data_width = self.data_width # hack to do with it not being initialised
        param = self.__dict__
        param["kernel_size"] = self.kernel_size
        param = namedtuple('ForkParam', param.keys())(*param.values())

        # fold the depth dimension into the col dimension
        param._replace(cols=param.cols * param.depth)

        # call the 2D utilisation model instead
        return Fork.utilisation_model(param)

    def get_pred_array(self):

        # load utilisation model from the 2D model
        self.data_width = self.data_width # hack to do with it not being initialised
        param = namedtuple('ForkParam', self.__dict__.keys())(*self.__dict__.values())

        # fold the depth dimension into the col dimension
        param._replace(cols=param.cols * param.depth)

        # call the 2D utilisation model instead
        return Fork.get_pred_array(param)

    def visualise(self, name):
        return pydot.Node(name,label="fork3d", shape="box",
                style="filled", fillcolor="azure",
                fontsize=MODULE_3D_FONTSIZE)

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth    , "ERROR: invalid depth dimension"
        assert data.shape[3] == self.channels, "ERROR: invalid channel dimension"
        assert data.shape[4] == self.kernel_rows  , "ERROR: invalid kernel row dimension"
        assert data.shape[5] == self.kernel_cols  , "ERROR: invalid kernel column dimension"
        assert data.shape[6] == self.kernel_depth  , "ERROR: invalid kernel depth dimension"

        out = np.ndarray((
            self.rows,
            self.cols,
            self.depth,
            self.channels,
            self.coarse,
            self.kernel_rows,
            self.kernel_cols,
            self.kernel_depth),dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = data[
              index[0],
              index[1],
              index[2],
              index[3],
              index[5],
              index[6],
              index[7]]

        return out

