"""
"""

import math
import os
import sys
from dataclasses import dataclass, field
from collections import namedtuple

import numpy as np
import pydot

from fpgaconvnet.models.modules import int2bits, Module3D, MODULE_3D_FONTSIZE
from fpgaconvnet.tools.resource_analytical_model import dsp_multiplier_resource_model
from fpgaconvnet.tools.resource_analytical_model import bram_memory_resource_model

from fpgaconvnet.models.modules import GlobalPool

@dataclass
class GlobalPool3D(Module3D):
    backend: str = "chisel"
    acc_width: int = 32

    def __post_init__(self):

        # get the module identifer
        self.module_identifier = "GlobalPool"

        # load resource coefficients
        self.load_resource_coefficients(self.module_identifier)

    def utilisation_model(self):

        # load utilisation model from the 2D model
        self.data_width = self.data_width # hack to do with it not being initialised
        param = namedtuple('GlobalPoolParam', self.__dict__.keys())(*self.__dict__.values())

        # fold the depth dimension into the col dimension
        param._replace(cols=param.cols * param.depth)

        # call the 2D utilisation model instead
        return GlobalPool.utilisation_model(param)

    def depth_out(self):
        return 1

    def rows_out(self):
        return 1

    def cols_out(self):
        return 1

    def module_info(self):#TODO
        # get the base module fields
        info = Module3D.module_info(self)
        # return the info
        return info

    def rsc(self,coef=None):
        # use module resource coefficients if none are given
        if coef == None:
            coef = self.rsc_coef
        # get the average polling buffer BRAM estimate
        avgpool_buffer_bram = bram_memory_resource_model(int(self.channels), self.data_width)

        # get the linear model estimation
        rsc = Module3D.rsc(self, coef)

        # add the bram estimation
        rsc["BRAM"] = avgpool_buffer_bram

        # add the dsp estimation
        rsc["DSP"] = 1

        # return the resource usage
        return rsc

    def visualise(self, name):
        return pydot.Node(name, label="global_pool3d", shape="box",
                style="filled", fillcolor="chartreuse",
                fontsize=MODULE_3D_FONTSIZE)


    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows       , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols       , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth      , "ERROR: invalid depth dimension"
        assert data.shape[3] == self.channels   , "ERROR: invalid channel dimension"

        # return average
        return np.average(data, axis=(0,1,2))
