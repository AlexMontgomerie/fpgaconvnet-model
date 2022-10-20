"""
The Glue module is used to combine streams
used for channel parallelism in the
Convolution layer together.

.. figure:: ../../../figures/glue_diagram.png
"""
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pydot

from fpgaconvnet.models.modules import Module3D, MODULE_3D_FONTSIZE

@dataclass
class Glue3D(Module3D):
    filters: int
    coarse_in: int
    coarse_out: int
    acc_width: int = field(default=16, init=False)

    def __post_init__(self):
        pass
        # load the resource model coefficients
        self.rsc_coef["LUT"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/glue_lut.npy"))
        self.rsc_coef["FF"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/glue_ff.npy"))
        self.rsc_coef["BRAM"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/glue_bram.npy"))
        self.rsc_coef["DSP"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/glue_dsp.npy"))

    def utilisation_model(self):
        pass
        return {
            "LUT"   : np.array([self.cols,self.rows,self.channels,self.data_width,self.acc_width,self.filters,self.coarse_in,self.coarse_out]),
            "FF"    : np.array([self.cols,self.rows,self.channels,self.data_width,self.acc_width,self.filters,self.coarse_in,self.coarse_out]),
            "DSP"   : np.array([self.cols,self.rows,self.channels,self.data_width,self.acc_width,self.filters,self.coarse_in,self.coarse_out]),
            "BRAM"  : np.array([self.cols,self.rows,self.channels,self.data_width,self.acc_width,self.filters,self.coarse_in,self.coarse_out]),
        }

    def channels_in(self):
        pass
        return self.filters

    def channels_out(self):
        pass
        return self.filters

    def get_latency(self):
        pass
        return self.rows *self.cols *self.filters / self.coarse_out

    def module_info(self):
        pass
        # get the base module fields
        info = Module3D.module_info(self)
        # add module-specific info fields
        info["filters"] = self.filters
        info["coarse_in"] = self.coarse_in
        info["coarse_out"] = self.coarse_out
        # return the info
        return info

    def visualise(self, name):
        pass
        return pydot.Node(name,label="glue", shape="box",
                style="filled", fillcolor="fuchsia",
                fontsize=MODULE_3D_FONTSIZE)

    def functional_model(self,data):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth    , "ERROR: invalid depth dimension"
        assert data.shape[3] == int(self.filters/self.coarse_out) , "ERROR: invalid  dimension"
        assert data.shape[4] == self.coarse_in , "ERROR: invalid  dimension"
        assert data.shape[5] == self.coarse_out , "ERROR: invalid  dimension"

        out = np.zeros((
            self.rows,
            self.cols,
            self.depth,
            int(self.filters/self.coarse_out),
            self.coarse_out),dtype=float)

        for index,_ in np.ndenumerate(out):
            for c in range(self.coarse_in):
                out[index] += data[index[0],index[1],index[2],index[3],c,index[4]]

        return out

