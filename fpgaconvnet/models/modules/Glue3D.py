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

from fpgaconvnet.models.modules import int2bits, Module3D, MODULE_3D_FONTSIZE

@dataclass
class Glue3D(Module3D):
    filters: int
    coarse_in: int
    coarse_out: int
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
        return self.filters

    def channels_out(self):
        return self.filters

    def get_latency(self):
        return self.rows * self.cols * self.depth * self.filters / self.coarse_out

    def module_info(self):
        # get the base module fields
        info = Module3D.module_info(self)
        # add module-specific info fields
        info["filters"] = self.filters
        info["coarse_in"] = self.coarse_in
        info["coarse_out"] = self.coarse_out
        # return the info
        return info

    def utilisation_model(self):
        if self.backend == "hls":
            pass
        elif self.backend == "chisel":
            return {
                "Logic_LUT" : np.array([
                    self.data_width*self.coarse_in, # tree buffer
                    self.data_width*int2bits(self.coarse_in), # tree buffer
                    self.coarse_in, # input ready
                    1,
                ]),
                "LUT_RAM" : np.array([
                    self.data_width*(int2bits(self.coarse_in)+1), # tree buffer
                    1,
                ]),
                "LUT_SR" : np.array([
                    int2bits(self.coarse_in), # tree buffer valid
                    1,
                ]),
                "FF" : np.array([
                    self.data_width, # output buffer
                    int2bits(self.coarse_in), # tree buffer valid
                    int2bits(max(1,int2bits(self.coarse_in))), # tree buffer queue buffer
                    # self.coarse_in, # ready signal
                    self.data_width*(self.coarse_in + math.floor((self.coarse_in-5)/2)), # adder tree reg
                    1,
                ]),
                "DSP"       : np.array([0]),
                "BRAM36"    : np.array([0]),
                "BRAM18"    : np.array([0]),
            }
        else:
            raise ValueError(f"{self.backend} backend not supported")

    def visualise(self, name):
        return pydot.Node(name,label="glue3d", shape="box",
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

