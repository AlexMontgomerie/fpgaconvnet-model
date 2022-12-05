"""
"""

import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pydot

from fpgaconvnet.models.modules import int2bits, Module3D, MODULE_3D_FONTSIZE
from fpgaconvnet.tools.resource_analytical_model import dsp_multiplier_resource_model
from fpgaconvnet.tools.resource_analytical_model import bram_memory_resource_model

@dataclass
class GlobalPool3D(Module3D):
    backend: str = "chisel"
    acc_width: int = field(default=32, init=False)

    def __post_init__(self):

        # get the cache path
        rsc_cache_path = os.path.dirname(__file__) + \
                f"/../../coefficients/{self.backend}"

        # iterate over resource types
        self.rsc_coef = {}
        for rsc_type in self.utilisation_model():
            # load the resource coefficients from the 2D version
            # coef_path = os.path.join(rsc_cache_path, f"{self.__class__.__name__.split('3D')[0]}_{rsc_type}.npy".lower())
            coef_path = os.path.join(rsc_cache_path, f"{self.__class__.__name__.replace('Global', 'Average').split('3D')[0]}_{rsc_type}.npy".lower()) #TODO: fix this hack
            self.rsc_coef[rsc_type] = np.load(coef_path)

    def utilisation_model(self):

        if self.backend == "hls":
            raise NotImplementedError
        elif self.backend == "chisel":
            return {
                "Logic_LUT" : np.array([
                    self.acc_width, # adder
                    self.data_width, # adder
                    int2bits(self.channels), # channel_cntr
                    int2bits(self.rows*self.cols*self.depth), # spatial cntr
                    self.channels, # acc logic
                    1,
                ]),
                "LUT_RAM"   : np.array([
                    self.data_width, # output queue
                    self.acc_width*self.channels,
                ]),
                "LUT_SR"    : np.array([0]),
                "FF"        : np.array([
                    self.data_width, # input cache
                    int2bits(self.channels), # channel_cntr
                    int2bits(self.rows*self.cols*self.depth), # spatial cntr
                    self.acc_width*self.channels, # accumulation reg
                    1, # other registers
                ]),
                "DSP"       : np.array([0]),
                "BRAM36"    : np.array([0]),
                "BRAM18"    : np.array([0]),
            }

        else:
            raise NotImplementedError(f"{self.backend} backend not supported")

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
