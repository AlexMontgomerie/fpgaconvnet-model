"""
"""

import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pydot

from fpgaconvnet.models.modules import Module, MODULE_FONTSIZE
from fpgaconvnet.tools.resource_analytical_model import dsp_multiplier_resource_model

@dataclass
class AveragePool(Module):

    def __post_init__(self):
        return
        # load the resource model coefficients
        #TODO add model coefs FOR BIAS - currently using conv to approx.
        # load the resource model coefficients
        # self.rsc_coef["LUT"] = np.load(
        #         os.path.join(os.path.dirname(__file__),
        #         "../../coefficients/accum_lut.npy"))
        # self.rsc_coef["FF"] = np.load(
        #         os.path.join(os.path.dirname(__file__),
        #         "../../coefficients/accum_ff.npy"))
        # self.rsc_coef["BRAM"] = np.load(
        #         os.path.join(os.path.dirname(__file__),
        #         "../../coefficients/accum_bram.npy"))
        # self.rsc_coef["DSP"] = np.load(
        #         os.path.join(os.path.dirname(__file__),
        #         "../../coefficients/accum_dsp.npy"))

    def utilisation_model(self):#TODO - copied from acum, FIXME
        return {
            "LUT"   : np.array([1]),
            "FF"    : np.array([1]),
            "DSP"   : np.array([1]),
            "BRAM"  : np.array([1]),
        }

    def rows_out(self):
        return 1

    def cols_out(self):
        return 1

    def module_info(self):#TODO
        # get the base module fields
        info = Module.module_info(self)
        # return the info
        return info

    def rsc(self,coef=None):#TODO replace conv version of func
        # get the linear model estimation
        # rsc = Module.rsc(self, coef)
        # # return the resource model
        # return rsc
        return {
            "LUT"   : 1,
            "FF"    : 1,
            "DSP"   : 0,
            "BRAM"  : 0,
        }

    def visualise(self, name):
        return pydot.Node(name, label="average_pool", shape="box",
                style="filled", fillcolor="chartreuse",
                fontsize=MODULE_FONTSIZE)


    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows       , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols       , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels   , "ERROR: invalid channel dimension"

        # return average
        return np.average(data, axis=(0,1))


