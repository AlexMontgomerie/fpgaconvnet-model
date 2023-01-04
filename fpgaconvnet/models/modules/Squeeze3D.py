import math
import os
import sys
from dataclasses import dataclass, field
from collections import namedtuple

import pydot
import numpy as np

from fpgaconvnet.models.modules import int2bits, Module3D, MODULE_3D_FONTSIZE

from fpgaconvnet.models.modules import Squeeze

@dataclass
class Squeeze3D(Module3D):
    coarse_in: int
    coarse_out: int
    backend: str = "chisel"
    regression_model: str = "linear_regression"
    streams: int = 1
    latency_mode: int = False
    block: int = False

    def __post_init__(self):

        # get the module identifer
        self.module_identifier = "Squeeze"

        # load resource coefficients
        self.load_resource_coefficients(self.module_identifier)

    def module_info(self):
        # get the base module fields
        info = Module3D.module_info(self)
        # add module-specific info fields
        info["coarse_in"] = self.coarse_in
        info["coarse_out"] = self.coarse_out
        # return the info
        return info

    @staticmethod
    def lcm(a, b):
        return abs(a*b) // math.gcd(a, b)

    def memory_usage(self):
        buffer_size = self.lcm(self.coarse_in, self.coarse_out)
        if self.backend == "chisel":
            return self.data_width*buffer_size*((buffer_size//self.coarse_in)+1) # buffer
        else:
            raise NotImplementedError

    def utilisation_model(self):

        # load utilisation model from the 2D model
        self.data_width = self.data_width # hack to do with it not being initialised
        param = namedtuple('SqueezeParam', self.__dict__.keys())(*self.__dict__.values())

        # fold the depth dimension into the col dimension
        param._replace(cols=param.cols * param.depth)

        # call the 2D utilisation model instead
        return Squeeze.utilisation_model(param)

    def get_pred_array(self):

        # load utilisation model from the 2D model
        self.data_width = self.data_width # hack to do with it not being initialised
        param = namedtuple('SqueezeParam', self.__dict__.keys())(*self.__dict__.values())

        # call the 2D utilisation model instead
        return Squeeze.get_pred_array(param)

    def visualise(self, name):
        distortion = 0
        if self.coarse_in > self.coarse_out:
            distortion = self.coarse_in/self.coarse_out
        else:
            distortion = -self.coarse_out/self.coarse_in
        return pydot.Node(name,label="squeeze3d", shape="polygon",
                sides=4, distortion=distortion, style="filled",
                fillcolor="olive", fontsize=MODULE_3D_FONTSIZE)

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows                       , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols                       , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth                      , "ERROR: invalid depth dimension"
        assert data.shape[3] == self.channels//self.coarse_in   , "ERROR: invalid channel dimension"
        assert data.shape[4] == self.coarse_in                  , "ERROR: invalid coarse dimension"

        out = np.ndarray((
            self.rows,
            self.cols,
            self.depth,
            self.channels//self.coarse_out,
            self.coarse_out),dtype=float)

        out = np.reshape(data,(self.rows,self.cols,self.depth,self.channels))
        out = np.reshape(data,(self.rows,self.cols,self.depth,self.channels//self.coarse_out,self.coarse_out))

        return out


