"""
"""

import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pydot

from fpgaconvnet.models.modules import int2bits, Module, MODULE_FONTSIZE
from fpgaconvnet.tools.resource_analytical_model import dsp_multiplier_resource_model

@dataclass
class GlobalPool(Module):
    backend: str = "chisel"
    acc_width: int = field(default=32, init=False)

    def __post_init__(self):
        return

    def utilisation_model(self):

        if self.backend == "hls":
            raise NotImplementedError
        elif self.backend == "chisel":
            return {
                "Logic_LUT" : np.array([
                    self.acc_width, # adder
                    self.data_width, # adder
                    int2bits(self.channels), # channel_cntr
                    int2bits(self.rows*self.cols), # spatial cntr
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
                    int2bits(self.rows*self.cols), # spatial cntr
                    self.acc_width*self.channels, # accumulation reg
                    1, # other registers
                ]),
                "DSP"       : np.array([1]),
                "BRAM36"    : np.array([0]),
                "BRAM18"    : np.array([0]),
            }

        else:
            raise NotImplementedError(f"{self.backend} backend not supported")


    def rows_out(self):
        return 1

    def cols_out(self):
        return 1

    def module_info(self):
        # get the base module fields
        info = Module.module_info(self)
        # return the info
        return info

    # def rsc(self,coef=None):
    #     # use module resource coefficients if none are given
    #     if coef == None:
    #         coef = self.rsc_coef
    #     # get the linear model estimation
    #     rsc = Module.rsc(self, coef)
    #     # return the resource model
    #     return rsc

    def visualise(self, name):
        return pydot.Node(name, label="global_pool", shape="box",
                style="filled", fillcolor="chartreuse",
                fontsize=MODULE_FONTSIZE)


    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows       , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols       , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels   , "ERROR: invalid channel dimension"

        # return average
        return np.average(data, axis=(0,1))


