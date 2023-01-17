"""
The purpose of the accumulation (Accum) module is
to perform the channel-wise accumulation of the
dot product result from the output of the
convolution (Conv) module.  As the data is coming
into the module filter-first, the separate filter
accumulations are buffered until they complete
their accumulation across channels.

.. figure:: ../../../figures/accum_diagram.png
"""
import math
import os
import sys
from dataclasses import dataclass, field
from collections import namedtuple

import numpy as np
import pydot

from fpgaconvnet.models.modules import int2bits, Module3D, MODULE_3D_FONTSIZE
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model

from fpgaconvnet.models.modules import Accum

@dataclass
class Accum3D(Module3D):
    filters: int
    groups: int
    backend: str = "chisel"
    regression_model: str = "linear_regression"
    data_width: int = field(default=32, init=False)
    streams: int = 1
    latency_mode: int = False
    block: int = False

    def __post_init__(self):

        # get the module identifer
        self.module_identifier = "Accum"

        # load resource coefficients
        self.load_resource_coefficients(self.module_identifier)

    def channels_in(self):
        return (self.channels*self.filters)//self.groups

    def channels_out(self):
        return self.filters

    def rate_out(self):
        return (self.groups)/float(self.channels)

    def pipeline_depth(self):
        return (self.channels*self.filters)//(self.groups*self.groups)

    def module_info(self):
        # get the base module fields
        info = Module3D.module_info(self)
        # add module-specific info fields
        info['groups'] = self.groups
        info['filters'] = self.filters
        info['channels_per_group'] = self.channels_in()//self.groups
        info['filters_per_group'] = self.filters//self.groups
        # return the info
        return info

    def memory_usage(self):
        return int(self.filters/self.groups)*self.data_width

    def utilisation_model(self):

        # load utilisation model from the 2D model
        self.data_width = self.data_width # hack to do with it not being initialised
        param = namedtuple('AccumParam', self.__dict__.keys())(*self.__dict__.values())

        # fold the depth dimension into the col dimension
        param._replace(cols=param.cols * param.depth)

        # call the 2D utilisation model instead
        return Accum.utilisation_model(param)

    def get_pred_array(self):

        # load utilisation model from the 2D model
        self.data_width = self.data_width # hack to do with it not being initialised
        param = namedtuple('AccumParam', self.__dict__.keys())(*self.__dict__.values())

        # fold the depth dimension into the col dimension
        param._replace(cols=param.cols * param.depth)

        # call the 2D utilisation model instead
        return Accum.get_pred_array(param)

    def visualise(self, name):
        return pydot.Node(name,label="accum3d", shape="box",
                height=self.filters/self.groups*0.25,
                style="filled", fillcolor="coral",
                fontsize=MODULE_3D_FONTSIZE)

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows                   , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols                   , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth                  , "ERROR: invalid depth dimension"
        assert data.shape[3] == self.channels               , "ERROR: invalid channel dimension"
        assert data.shape[4] == self.filters//self.groups   , "ERROR: invalid filter  dimension"

        channels_per_group = self.channels//self.groups
        filters_per_group  = self.filters//self.groups

        out = np.zeros((
            self.rows,
            self.cols,
            self.depth,
            self.filters),dtype=float)

        tmp = np.zeros((
            self.rows,
            self.cols,
            self.depth,
            channels_per_group,
            filters_per_group),dtype=float)

        for index,_ in np.ndenumerate(tmp):
            for g in range(self.groups):
                out[index[0],index[1],index[2],g*filters_per_group+index[4]] = \
                        float(out[index[0],index[1],index[2],g*filters_per_group+index[4]]) + \
                        float(data[index[0],index[1],index[2],g*channels_per_group+index[3],index[4]])

        return out

