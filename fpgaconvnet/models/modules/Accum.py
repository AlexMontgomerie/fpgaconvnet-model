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

from fpgaconvnet.models.modules import Module, MODULE_FONTSIZE
from fpgaconvnet.tools.resource_model import bram_memory_resource_model

@dataclass
class Accum(Module):
    filters: int
    groups: int

    def __post_init__(self):
        # load the resource model coefficients
        self.rsc_coef["LUT"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/accum_lut.npy"))
        self.rsc_coef["FF"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/accum_ff.npy"))
        self.rsc_coef["BRAM"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/accum_bram.npy"))
        self.rsc_coef["DSP"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/accum_dsp.npy"))

    def utilisation_model(self):
        return {
            "LUT"   : np.array([self.filters,self.groups,self.data_width,self.cols,self.rows,self.channels]),
            "FF"    : np.array([self.filters,self.groups,self.data_width,self.cols,self.rows,self.channels]),
            "DSP"   : np.array([self.filters,self.groups,self.data_width,self.cols,self.rows,self.channels]),
            "BRAM"  : np.array([self.filters,self.groups,self.data_width,self.cols,self.rows,self.channels]),
        }

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
        info = Module.module_info(self)
        # add module-specific info fields
        info['groups'] = self.groups
        info['filters'] = self.filters
        info['channels_per_group'] = self.channels_in()//self.groups
        info['filters_per_group'] = self.filters//self.groups
        # return the info
        return info

    def rsc(self,coef=None):
        # use module resource coefficients if none are given
        if coef == None:
            coef = self.rsc_coef
        # get the accumulation buffer BRAM estimate
        acc_buffer_bram = bram_memory_resource_model(int(self.filters/self.groups), self.data_width)
        # get the linear model estimation
        rsc = Module.rsc(self, coef)
        # add the bram estimation
        rsc["BRAM"] = acc_buffer_bram
        # return the resource usage
        return rsc

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
                out[index[0],index[1],g*filters_per_group+index[3]] += data[index[0],index[1],g*channels_per_group+index[2],index[3]]

        return out

