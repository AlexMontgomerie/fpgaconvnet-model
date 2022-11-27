"""
"""

import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pydot

from fpgaconvnet.models.modules import Module3D, MODULE_3D_FONTSIZE
from fpgaconvnet.tools.resource_model import dsp_multiplier_resource_model
from fpgaconvnet.tools.resource_model import bram_memory_resource_model

@dataclass
class AveragePool3D(Module3D):

    def __post_init__(self):
        # load the resource model coefficients
        # TODO: Update resource model coefficients FIXME
        self.rsc_coef["LUT"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/avgpool3d_lut.npy"))
        self.rsc_coef["FF"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/avgpool3d_ff.npy"))
        self.rsc_coef["BRAM"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/avgpool3d_bram.npy"))
        self.rsc_coef["DSP"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/avgpool3d_dsp.npy"))

    def utilisation_model(self):
        # TODO: Update utilisation model FIXME
        return {
            "LUT"   : np.array([1]),
            "FF"    : np.array([1]),
            "DSP"   : np.array([1]),
            "BRAM"  : np.array([1]),
        }

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
        # return the resource usage
        return rsc

    def visualise(self, name):
        return pydot.Node(name, label="average_pool3d", shape="box",
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
