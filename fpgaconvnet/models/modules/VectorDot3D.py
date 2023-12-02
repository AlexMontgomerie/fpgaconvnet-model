import numpy as np
import math
import pydot
import os
from dataclasses import dataclass, field
from collections import namedtuple

from fpgaconvnet.models.modules import int2bits, Module3D, MODULE_3D_FONTSIZE
from fpgaconvnet.tools.resource_analytical_model import dsp_multiplier_resource_model

from fpgaconvnet.models.modules import VectorDot

@dataclass
class VectorDot3D(Module3D):
    filters: int
    fine: int
    backend: str = "chisel"
    regression_model: str = "linear_regression"
    weight_width: int = field(default=16, init=False)
    acc_width: int = field(default=32, init=False)
    streams: int = 1
    latency_mode: int = False
    block: int = False

    def __post_init__(self):

        # get the module identifer
        self.module_identifier = "VectorDot"

        # load resource coefficients
        self.load_resource_coefficients(self.module_identifier)

    def pipeline_depth(self):
        return int(math.log(self.fine, 2)) + 1

    def module_info(self):
        return {
            'type'      : self.__class__.__name__.upper(),
            'rows'      : self.rows_in(),
            'cols'      : self.cols_in(),
            'depth'      : self.depth_in(),
            'channels'  : self.channels_in(),
            'fine'    : self.fine,
            'filters'    : self.filters,
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'depth_out'      : self.depth_out(),
            'channels_out'  : self.channels_out()
        }

    def channels_out(self):
        return self.filters

    def rate_in(self):
        return 1.0/float(self.filters)

    def memory_usage(self):
        if self.backend == "chisel":
            return self.data_width*(int2bits(self.fine)+3)
        else:
            raise NotImplementedError

    def utilisation_model(self):

        # load utilisation model from the 2D model
        self.data_width = self.data_width # hack to do with it not being initialised
        self.weight_width = self.weight_width # hack to do with it not being initialised
        self.acc_width = self.acc_width # hack to do with it not being initialised
        param = namedtuple('VectorDotParam', self.__dict__.keys())(*self.__dict__.values())

        # fold the depth dimension into the col dimension
        param._replace(cols=param.cols * param.depth)

        # call the 2D utilisation model instead
        return VectorDot.utilisation_model(param)

    def get_pred_array(self):

        # load utilisation model from the 2D model
        self.data_width = self.data_width # hack to do with it not being initialised
        param = namedtuple('VectorDotParam', self.__dict__.keys())(*self.__dict__.values())

        # call the 2D utilisation model instead
        return VectorDot.get_pred_array(param)

    def rsc(self,coef=None, model=None):

        # get the linear model estimation
        rsc = Module3D.rsc(self, coef, model)

        if self.regression_model == "linear_regression":
            # get the dsp usage
            dsp = self.streams*self.fine*dsp_multiplier_resource_model(
                    self.data_width, self.data_width)
            rsc["DSP"] = dsp

        return rsc

    '''
    FUNCTIONAL MODEL
    '''

    def visualise(self, name):
        return pydot.Node(name,label="vector_dot3d", shape="box",
                          style="filled", fillcolor="gold",
                          fontsize=MODULE_3D_FONTSIZE)

    def functional_model(self, data, weights):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth   , "ERROR: invalid depth dimension"
        assert data.shape[3] == self.channels, "ERROR: invalid channel dimension"
        assert data.shape[4] == self.fine    , "ERROR: invalid column dimension"
        # check input dimensionality
        assert weights.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert weights.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert weights.shape[2] == self.depth   , "ERROR: invalid depth dimension"
        assert weights.shape[3] == self.channels, "ERROR: invalid channel dimension"
        assert weights.shape[4] == self.filters , "ERROR: invalid channel dimension"
        assert weights.shape[5] == self.fine    , "ERROR: invalid column dimension"


        out = np.ndarray((
            self.rows,
            self.cols,
            self.depth,
            self.channels,
            self.filters),dtype=object)

        for index,_ in np.ndenumerate(weights):
            if index[5] == 0:
                out[
                    index[0],
                    index[1],
                    index[2],
                    index[3],
                    index[4]
                ] = weights[index] * data[
                        index[0],
                        index[1],
                        index[2],
                        index[3],
                        index[5]
                    ]
            else:
                out[
                    index[0],
                    index[1],
                    index[2],
                    index[3],
                    index[4]
                ] += weights[index] * data[
                        index[0],
                        index[1],
                        index[2],
                        index[3],
                        index[5]
                    ]

        return out


