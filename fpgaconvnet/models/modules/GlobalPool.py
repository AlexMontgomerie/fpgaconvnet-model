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
from fpgaconvnet.tools.resource_analytical_model import queue_lutram_resource_model

@dataclass
class GlobalPool(Module):
    backend: str = "chisel"
    regression_model: str = "linear_regression"
    acc_width: int = field(default=32, init=False)

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
                    1,
                ]),
                "LUT_RAM"   : np.array([
                    queue_lutram_resource_model(
                        4, self.data_width), # buffer
                ]),
                "LUT_SR"    : np.array([0]),
                "FF"        : np.array([
                    self.data_width, # input cache
                    int2bits(self.channels), # channel_cntr
                    int2bits(self.rows*self.cols), # spatial cntr
                    self.acc_width*self.channels, # accumulation reg
                    1, # other registers
                ]),
                "DSP"       : np.array([0]),
                "BRAM36"    : np.array([0]),
                "BRAM18"    : np.array([0]),
            }

        else:
            raise NotImplementedError(f"{self.backend} backend not supported")

    def get_pred_array(self):
        return np.array([
        self.data_width, self.data_width//2,
        self.channels, self.rows, self.cols,
        self.acc_width, self.acc_width//2,
        ]).reshape(1,-1)

    def rsc(self, coef=None, model=None):

        # get the regression model estimation
        rsc = Module.rsc(self, coef, model)

        if self.regression_model == "linear_regression":
            # get the dsp usage
            rsc["DSP"] = dsp_multiplier_resource_model(
                    self.data_width, self.acc_width)

        return rsc

    def rows_out(self):
        return 1

    def cols_out(self):
        return 1

    def module_info(self):
        # get the base module fields
        info = Module.module_info(self)
        # return the info
        return info

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

    def pipeline_depth(self):
        return self.rows*self.cols*self.channels