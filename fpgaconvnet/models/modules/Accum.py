"""
The purpose of the accumulation (Accum) module is
to perform the channel-wise accumulation of the
dot product result from the output of the
convolution (Conv) module.  As the data is coming
into the module filter-first, the separate filter
accumulations are buffered until they complete
their accumulation across channels.

.. figure:: ../../../figures/accum_diagram.png
"""
import math
import os
import sys
from dataclasses import dataclass, field
import importlib

import numpy as np
import pydot

from fpgaconvnet.models.modules import int2bits, Module, MODULE_FONTSIZE
from fpgaconvnet.tools.resource_analytical_model import queue_lutram_resource_model

@dataclass
class Accum(Module):
    channels: int
    filters: int
    backend: str = "chisel"
    regression_model: str = "linear_regression"
    streams: int = 1

    @property
    def input_shape(self):
        return (self.channels, self.filters)

    @property
    def output_shape(self):
        return (self.filters)

    def rate_out(self):
        return 1.0/float(self.channels)

    def pipeline_depth(self):
        # return (self.channels*self.filters)//(self.groups*self.groups)
        return self.channels

    def module_info(self):
        # get the base module fields
        info = Module.module_info(self)
        # add module-specific info fields
        info['channels'] = self.channels
        info['filters'] = self.filters
        # return the info
        return info

    def utilisation_model(self):

        if self.backend == "hls":
            return {
                "LUT"   : np.array([
                    self.filters, self.data_t.width, self.channels
                ]),
                "FF"    : np.array([
                    self.filters, self.data_t.width, self.channels
                ]),
                "DSP"   : np.array([
                    self.filters, self.data_t.width, self.channels
                ]),
                "BRAM"  : np.array([
                    self.filters, self.data_t.width, self.channels
                ]),
            }

        elif self.backend == "chisel":
            return {
                "Logic_LUT" : np.array([
                    self.filters, self.channels, # parameter logic
                    self.streams*self.data_t.width, # input word logic
                    self.streams, # input streams logic
                    int2bits(self.channels), # channel cntr
                    int2bits(self.filters), # filter cntr
                    1, # extra
                ]),
                "LUT_RAM"   : np.array([
                    queue_lutram_resource_model(
                        2, self.streams*self.data_t.width), # output buffer
                    self.streams*self.data_t.width*self.filters, # filter memory memory (size)
                    self.streams*self.data_t.width, # filter memory memory (word width)
                    self.streams*self.filters, # filter memory memory (depth)
                ]),
                "LUT_SR"    : np.array([0]),
                "FF"        : np.array([
                    self.data_t.width,  # input val cache
                    self.streams*self.data_t.width,  # input val cache
                    int2bits(self.channels), # channel_cntr
                    int2bits(self.filters), # filter cntr
                    self.channels, # channel parameter reg
                    self.filters, # filter parameter reg
                    1, # other registers
                ]),
                "DSP"       : np.array([0]),
                "BRAM36"    : np.array([0]),
                "BRAM18"    : np.array([0]),
            }

        else:
            raise ValueError(f"{self.backend} backend not supported")

    def get_pred_array(self):
        return np.array([
            self.data_t.width, self.data_t.binary_point,
            self.channels, self.filters, self.streams
        ]).reshape(1,-1)

    def visualise(self, name):
        return pydot.Node(name,label="accum", shape="box",
                height=self.filters/self.groups*0.25,
                style="filled", fillcolor="coral",
                fontsize=MODULE_FONTSIZE)

    def functional_model(self, data):

        # check input dimensionality
        assert data.shape[0] == self.repeat     , "ERROR: invalid row dimension"
        assert data.shape[1] == self.channels   , "ERROR: invalid channel dimension"
        assert data.shape[2] == self.filters    , "ERROR: invalid filter  dimension"

        # return the accumulated channel dimension
        return np.reshape(np.sum(data, axis=1), (self.repeat, self.filters))

