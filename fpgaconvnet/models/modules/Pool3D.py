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
from collections import namedtuple

import numpy as np
import pydot

from fpgaconvnet.models.modules import Module3D, MODULE_3D_FONTSIZE

from fpgaconvnet.models.modules import Pool

@dataclass
class Pool3D(Module3D):
    # kernel_size: Union[List[int], int]
    kernel_rows: int
    kernel_cols: int
    kernel_depth: int
    pool_type: str = "max"
    backend: str = "chisel"
    regression_model: str = "linear_regression"
    streams: int = 1
    latency_mode: int = False
    block: int = False

    def __post_init__(self):

        # get the module identifer
        self.module_identifier = "Pool"

        # load resource coefficients
        self.load_resource_coefficients(self.module_identifier)

    @property
    def kernel_size(self):
        return [ self.kernel_rows, self.kernel_cols, self.kernel_depth ]

    def utilisation_model(self):

        # load utilisation model from the 2D model
        self.data_width = self.data_width # hack to do with it not being initialised
        param = self.__dict__
        param["kernel_size"] = self.kernel_size
        param = namedtuple('PoolParam', param.keys())(*param.values())

        # fold the depth dimension into the col dimension
        param._replace(cols=param.cols * param.depth)

        # call the 2D utilisation model instead
        return Pool.utilisation_model(param)

    def get_pred_array(self):

        # load utilisation model from the 2D model
        self.data_width = self.data_width # hack to do with it not being initialised
        param = namedtuple('PoolParam', self.__dict__.keys())(*self.__dict__.values())

        # fold the kernel depth dimension into the kernel col dimension
        param = param._replace(kernel_size=[self.kernel_rows, self.kernel_cols * self.kernel_depth])

        # call the 2D utilisation model instead
        return Pool.get_pred_array(param)

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

