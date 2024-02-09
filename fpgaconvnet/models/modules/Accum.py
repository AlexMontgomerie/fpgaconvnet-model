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
from typing import List

from fpgaconvnet.models.modules import int2bits, Module, MODULE_FONTSIZE
from fpgaconvnet.tools.resource_analytical_model import queue_lutram_resource_model

@dataclass
class Accum(Module):
    filters: int
    groups: int
    skip_all_zero_window: bool = False
    sparsity_hist: np.ndarray = None
    backend: str = "chisel"
    regression_model: str = "linear_regression"
    streams: int = 1

    def channels_in(self):
        return (self.channels*self.filters)//self.groups

    def channels_out(self):
        return self.filters

    def rate_sparsity(self):
        return np.min(1.0/(1-np.array(self.sparsity_hist[:, -1])))

    def rate_in(self):
        if self.skip_all_zero_window:
            return self.rate_sparsity()
        else:
            return super().rate_in()

    def rate_out(self):
        if self.skip_all_zero_window:
            return min(float(self.channels), self.rate_sparsity())/float(self.channels) * (self.groups)
        else:
            return (self.groups)/float(self.channels)

    def pipeline_depth(self):
        return (self.channels*self.filters)//(self.groups*self.groups)
        # return self.channels//self.groups
        # return self.filters*(self.channels-1)//(self.groups*self.groups)

    def module_info(self):
        # get the base module fields
        info = Module.module_info(self)
        # add module-specific info fields
        info['groups'] = self.groups
        info['filters'] = self.filters
        info['channels_per_group'] = self.channels_in()//self.groups
        info['filters_per_group'] = self.filters//self.groups
        # return the info
        return info

    def memory_usage(self):
        return int(self.filters/self.groups)*self.data_width

    def utilisation_model(self):

        if self.backend == "hls":
            return {
                "LUT"   : np.array([
                    self.filters,self.groups,self.data_width,
                    self.cols,self.rows,self.channels
                ]),
                "FF"    : np.array([
                    self.filters,self.groups,self.data_width,
                    self.cols,self.rows,self.channels
                ]),
                "DSP"   : np.array([
                    self.filters,self.groups,self.data_width,
                    self.cols,self.rows,self.channels
                ]),
                "BRAM"  : np.array([
                    self.filters,self.groups,self.data_width,
                    self.cols,self.rows,self.channels
                ]),
            }

        elif self.backend == "chisel":
            return {
                "Logic_LUT" : np.array([
                    self.filters, self.channels, # parameter logic
                    self.streams*self.data_width, # input word logic
                    self.streams, # input streams logic
                    int2bits(self.channels), # channel cntr
                    int2bits(self.filters), # filter cntr
                    1, # extra
                ]),
                "LUT_RAM"   : np.array([
                    queue_lutram_resource_model(
                        2, self.streams*self.data_width), # output buffer
                    self.streams*self.data_width*self.filters, # filter memory memory (size)
                    self.streams*self.data_width, # filter memory memory (word width)
                    self.streams*self.filters, # filter memory memory (depth)
                ]),
                "LUT_SR"    : np.array([0]),
                "FF"        : np.array([
                    self.data_width,  # input val cache
                    self.streams*self.data_width,  # input val cache
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
            self.data_width, self.data_width//2,
            self.channels, self.filters, self.streams
        ]).reshape(1,-1)

    def visualise(self, name):
        return pydot.Node(name,label="accum", shape="box",
                height=self.filters/self.groups*0.25,
                style="filled", fillcolor="coral",
                fontsize=MODULE_FONTSIZE)

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows                   , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols                   , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels               , "ERROR: invalid channel dimension"
        assert data.shape[3] == self.filters//self.groups   , "ERROR: invalid filter  dimension"

        channels_per_group = self.channels//self.groups
        filters_per_group  = self.filters//self.groups

        out = np.zeros((
            self.rows,
            self.cols,
            self.filters),dtype=float)

        tmp = np.zeros((
            self.rows,
            self.cols,
            channels_per_group,
            filters_per_group),dtype=float)

        for index,_ in np.ndenumerate(tmp):
            for g in range(self.groups):
                out[index[0],index[1],g*filters_per_group+index[3]] = \
                        float(out[index[0],index[1],g*filters_per_group+index[3]]) + \
                        float(data[index[0],index[1],g*channels_per_group+index[2],index[3]])

        return out

