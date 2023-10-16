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
from collections import namedtuple

import numpy as np
import pydot

from fpgaconvnet.models.modules import Module3D, MODULE_3D_FONTSIZE
from fpgaconvnet.tools.resource_analytical_model import dsp_multiplier_resource_model

from fpgaconvnet.models.modules import Bias

@dataclass
class Bias3D(Module3D):
    filters: int
    biases_width: int = field(default=16, init=False)
    backend: str = "chisel"
    regression_model: str = "linear_regression"
    streams: int = 1

    def __post_init__(self):

        # get the module identifer
        self.module_identifier = "Bias"

        # load resource coefficients
        self.load_resource_coefficients(self.module_identifier)

    def channels_in(self):
        return self.filters

    def channels_out(self):
        return self.filters

    def module_info(self):#TODO
        # get the base module fields
        info = Module3D.module_info(self)
        # add module-specific info fields

        info['filters'] = self.filters

        # return the info
        return info

    def utilisation_model(self):

        # load utilisation model from the 2D model
        self.data_width = self.data_width # hack to do with it not being initialised
        param = namedtuple('BiasParam', self.__dict__.keys())(*self.__dict__.values())

        # fold the depth dimension into the col dimension
        param._replace(cols=param.cols * param.depth)

        # call the 2D utilisation model instead
        return Bias.utilisation_model(param)

    def visualise(self, name):
        return pydot.Node(name,label="bias3d", shape="box",
                style="filled", fillcolor="chartreuse",
                fontsize=MODULE_3D_FONTSIZE)

    def get_pred_array(self):

        # load utilisation model from the 2D model
        self.data_width = self.data_width # hack to do with it not being initialised
        param = namedtuple('BiasParam', self.__dict__.keys())(*self.__dict__.values())

        # fold the depth dimension into the col dimension
        param._replace(cols=param.cols * param.depth)

        # call the 2D utilisation model instead
        return Bias.get_pred_array(param)

    def functional_model(self,data,biases):
        # check input dimensionality
        assert data.shape[0] == self.rows                   , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols                   , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth                  , "ERROR: invalid depth dimension"
        assert data.shape[3] == self.filters                , "ERROR: invalid filter dimension"
        # check bias dimensionality
        assert biases.shape[0] == self.filters              , "ERROR: invalid filter dimension"

        out = np.zeros((
            self.rows,
            self.cols,
            self.depth,
            self.filters,
            ), dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = data[index] + biases[index[3]]

        # sanity check because numpy indexing confuses me
        for f_i in range(self.filters):
            # create copy of input and output filter
            cf = np.empty_like(data[:,:,:,0])
            cfo = np.empty_like(data[:,:,:,0])
            # set values of input and output
            cf[:] = data[:,:,:,f_i]
            cfo[:] = out[:,:,:,f_i]
            # subtraction should give bias
            v = cfo - cf
            for _,val in np.ndenumerate(v):
                # check each filter result has been added correctly to the bias
                assert np.allclose(biases[f_i],val,
                        rtol=1.e-8,atol=1.e-8), "ERROR: the biases don't match!"
        return out
