"""
The Glue module is used to combine streams
used for channel parallelism in the
Convolution layer together.

.. figure:: ../../../figures/glue_diagram.png
"""
import math
import os
import sys
from dataclasses import dataclass, field
from collections import namedtuple

import numpy as np
import pydot

from fpgaconvnet.models.modules import int2bits, Module3D, MODULE_3D_FONTSIZE

from fpgaconvnet.models.modules import Glue

@dataclass
class Glue3D(Module3D):
    filters: int
    coarse_in: int
    coarse_out: int
    backend: str = "chisel"
    regression_model: str = "linear_regression"
    data_width: int = field(default=32, init=False)
    streams: int = 1
    latency_mode: int = False
    block: int = False

    def __post_init__(self):

        # get the module identifer
        self.module_identifier = "Glue"

        # load resource coefficients
        self.load_resource_coefficients(self.module_identifier)

    def pipeline_depth(self):
        return self.coarse_in

    def channels_in(self):
        return self.filters

    def channels_out(self):
        return self.filters

    def latency(self):
        return self.rows * self.cols * self.depth * self.filters / self.coarse_out

    def module_info(self):
        # get the base module fields
        info = Module3D.module_info(self)
        # add module-specific info fields
        info["filters"] = self.filters
        info["coarse_in"] = self.coarse_in
        info["coarse_out"] = self.coarse_out
        # return the info
        return info

    def utilisation_model(self):

        # load utilisation model from the 2D model
        self.data_width = self.data_width # hack to do with it not being initialised
        param = namedtuple('GlueParam', self.__dict__.keys())(*self.__dict__.values())

        # fold the depth dimension into the col dimension
        param._replace(cols=param.cols * param.depth)

        # call the 2D utilisation model instead
        return Glue.utilisation_model(param)

    def get_pred_array(self):

        # load utilisation model from the 2D model
        self.data_width = self.data_width # hack to do with it not being initialised
        param = namedtuple('GlueParam', self.__dict__.keys())(*self.__dict__.values())

        # fold the depth dimension into the col dimension
        param._replace(cols=param.cols * param.depth)

        # call the 2D utilisation model instead
        return Glue.get_pred_array(param)

    def visualise(self, name):
        return pydot.Node(name,label="glue3d", shape="box",
                style="filled", fillcolor="fuchsia",
                fontsize=MODULE_3D_FONTSIZE)

    def functional_model(self,data):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth    , "ERROR: invalid depth dimension"
        assert data.shape[3] == int(self.filters/self.coarse_out) , "ERROR: invalid  dimension"
        assert data.shape[4] == self.coarse_in , "ERROR: invalid  dimension"
        assert data.shape[5] == self.coarse_out , "ERROR: invalid  dimension"

        out = np.zeros((
            self.rows,
            self.cols,
            self.depth,
            int(self.filters/self.coarse_out),
            self.coarse_out),dtype=float)

        for index,_ in np.ndenumerate(out):
            for c in range(self.coarse_in):
                out[index] += data[index[0],index[1],index[2],index[3],c,index[4]]

        return out

