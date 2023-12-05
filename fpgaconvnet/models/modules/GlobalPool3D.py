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

from fpgaconvnet.models.modules import GlobalPool

@dataclass
class GlobalPool3D(Module3D):
    backend: str = "chisel"
    regression_model: str = "linear_regression"
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

    def get_pred_array(self):

        # load utilisation model from the 2D model
        self.data_width = self.data_width # hack to do with it not being initialised
        param = namedtuple('GlobalPoolParam', self.__dict__.keys())(*self.__dict__.values())

        # fold the depth dimension into the col dimension
        param._replace(cols=param.cols * param.depth)

        # call the 2D utilisation model instead
        return GlobalPool.get_pred_array(param)

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

    def rsc(self, coef=None, model=None):

        # get the regression model estimation
        rsc = Module3D.rsc(self, coef, model)

        if self.regression_model == "linear_regression":
            # get the dsp usage
            rsc["DSP"] = dsp_multiplier_resource_model(
                    self.data_width, self.acc_width)

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

    def pipeline_depth(self):
        return self.rows*self.cols*self.depth*self.channels
