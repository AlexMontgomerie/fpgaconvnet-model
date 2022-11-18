import math
import os
import sys
from dataclasses import dataclass, field

import pydot
import numpy as np

from fpgaconvnet.models.modules import int2bits, Module, MODULE_FONTSIZE

def lcm(a, b):
    return abs(a*b) // math.gcd(a, b)


@dataclass
class Squeeze(Module):
    coarse_in: int
    coarse_out: int
    backend: str = "chisel"

    def module_info(self):
        # get the base module fields
        info = Module.module_info(self)
        # add module-specific info fields
        info["coarse_in"] = self.coarse_in
        info["coarse_out"] = self.coarse_out
        # return the info
        return info

    def utilisation_model(self):

        if self.backend == "hls":
            pass # TODO
        elif self.backend == "chisel":
            buffer_size = lcm(self.coarse_in, self.coarse_out)
            return {
                "Logic_LUT" : np.array([
                    (buffer_size//self.coarse_in), # buffer ready
                    self.data_width*self.coarse_out*(buffer_size//self.coarse_out), # arbiter logic
                    (buffer_size//self.coarse_in),
                    (buffer_size//self.coarse_out),
                    self.coarse_in,
                    self.coarse_out,
                    1,
                ]),
                "LUT_RAM"   : np.array([
                    self.data_width*buffer_size*((buffer_size//self.coarse_in)+1), # buffer
                    self.data_width*self.coarse_out, # output buffer
                    # 1,
                ]),
                "LUT_SR"    : np.array([0]),
                "FF"        : np.array([
                    (buffer_size//self.coarse_in), # cntr_in
                    self.coarse_out*int2bits(buffer_size//self.coarse_out), # arbiter registers
                    1,

                ]),
                "DSP"       : np.array([0]),
                "BRAM36"    : np.array([0]),
                "BRAM18"    : np.array([0]),
            }
        else:
            raise ValueError(f"{self.backend} backend not supported")

    def visualise(self, name):
        distortion = 0
        if self.coarse_in > self.coarse_out:
            distortion = self.coarse_in/self.coarse_out
        else:
            distortion = -self.coarse_out/self.coarse_in
        return pydot.Node(name,label="squeeze", shape="polygon",
                sides=4, distortion=distortion, style="filled",
                fillcolor="olive", fontsize=MODULE_FONTSIZE)

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows                       , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols                       , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels//self.coarse_in   , "ERROR: invalid channel dimension"
        assert data.shape[3] == self.coarse_in                  , "ERROR: invalid coarse dimension"

        out = np.ndarray((
            self.rows,
            self.cols,
            self.channels//self.coarse_out,
            self.coarse_out),dtype=float)

        out = np.reshape(data,(self.rows,self.cols,self.channels))
        out = np.reshape(data,(self.rows,self.cols,self.channels//self.coarse_out,self.coarse_out))

        return out


