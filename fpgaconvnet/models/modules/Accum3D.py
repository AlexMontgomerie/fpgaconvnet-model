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

import numpy as np
import pydot

from fpgaconvnet.models.modules import int2bits, Module3D, MODULE_3D_FONTSIZE
from fpgaconvnet.tools.resource_analytical_model import bram_memory_resource_model

@dataclass
class Accum3D(Module3D):
    filters: int
    groups: int
    backend: str = "chisel"

    def __post_init__(self):

        # get the cache path
        rsc_cache_path = os.path.dirname(__file__) + \
                f"/../../coefficients/{self.backend}"

        # iterate over resource types
        self.rsc_coef = {}
        for rsc_type in self.utilisation_model():
            # load the resource coefficients from the 2D version
            coef_path = os.path.join(rsc_cache_path, f"{self.__class__.__name__.split('3D')[0]}_{rsc_type}.npy".lower())
            self.rsc_coef[rsc_type] = np.load(coef_path)

    def channels_in(self):
        return (self.channels*self.filters)//self.groups

    def channels_out(self):
        return self.filters

    def rate_out(self):
        return (self.groups)/float(self.channels)

    def pipeline_depth(self):
        return (self.channels*self.filters)//(self.groups*self.groups)

    def module_info(self):
        # get the base module fields
        info = Module3D.module_info(self)
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
                    self.cols,self.rows,self.depth,self.channels
                ]),
                "FF"    : np.array([
                    self.filters,self.groups,self.data_width,
                    self.cols,self.rows,self.depth,self.channels
                ]),
                "DSP"   : np.array([
                    self.filters,self.groups,self.data_width,
                    self.cols,self.rows,self.depth,self.channels
                ]),
                "BRAM"  : np.array([
                    self.filters,self.groups,self.data_width,
                    self.cols,self.rows,self.depth,self.channels
                ]),
            }

        elif self.backend == "chisel":
            return {
                "Logic_LUT" : np.array([
                    self.filters, self.channels,
                    self.data_width, int2bits(self.channels),
                    int2bits(self.filters), 1,
                ]),
                "LUT_RAM"   : np.array([
                    self.data_width*self.filters, # output queue and memory
                    self.data_width, # acc buffer
                    1,
                ]),
                "LUT_SR"    : np.array([0]),
                "FF"        : np.array([
                    self.data_width,  # input val cache
                    int2bits(self.channels), # channel_cntr
                    int2bits(self.filters), # filter cntr
                    self.filters, # output queue and memory
                    1, # other registers
                ]),
                "DSP"       : np.array([0]),
                "BRAM36"    : np.array([0]),
                "BRAM18"    : np.array([0]),
            }

        else:
            raise ValueError(f"{self.backend} backend not supported")

    def rsc(self,coef=None):
        # use module resource coefficients if none are given
        if coef == None:
            coef = self.rsc_coef
        # get the accumulation buffer BRAM estimate
        # acc_buffer_bram = bram_memory_resource_model(int(self.filters/self.groups), self.data_width)

        # get the linear model estimation
        rsc = Module3D.rsc(self, coef)

        # add the bram estimation
        rsc["BRAM"] = 0

        # ensure zero DSPs
        rsc["DSP"] = 0

        # return the resource usage
        return rsc

    def visualise(self, name):
        return pydot.Node(name,label="accum3d", shape="box",
                height=self.filters/self.groups*0.25,
                style="filled", fillcolor="coral",
                fontsize=MODULE_3D_FONTSIZE)

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows                   , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols                   , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth                  , "ERROR: invalid depth dimension"
        assert data.shape[3] == self.channels               , "ERROR: invalid channel dimension"
        assert data.shape[4] == self.filters//self.groups   , "ERROR: invalid filter  dimension"

        channels_per_group = self.channels//self.groups
        filters_per_group  = self.filters//self.groups

        out = np.zeros((
            self.rows,
            self.cols,
            self.depth,
            self.filters),dtype=float)

        tmp = np.zeros((
            self.rows,
            self.cols,
            self.depth,
            channels_per_group,
            filters_per_group),dtype=float)

        for index,_ in np.ndenumerate(tmp):
            for g in range(self.groups):
                out[index[0],index[1],index[2],g*filters_per_group+index[4]] = \
                        float(out[index[0],index[1],index[2],g*filters_per_group+index[4]]) + \
                        float(data[index[0],index[1],index[2],g*channels_per_group+index[3],index[4]])

        return out

