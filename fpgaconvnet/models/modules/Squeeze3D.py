import math
import os
import sys
from dataclasses import dataclass, field

import pydot
import numpy as np

from fpgaconvnet.models.modules import int2bits, Module3D, MODULE_3D_FONTSIZE

@dataclass
class Squeeze3D(Module3D):
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

    def module_info(self):
        # get the base module fields
        info = Module3D.module_info(self)
        # add module-specific info fields
        info["coarse_in"] = self.coarse_in
        info["coarse_out"] = self.coarse_out
        # return the info
        return info

    @staticmethod
    def lcm(a, b):
        return abs(a*b) // math.gcd(a, b)

    def utilisation_model(self):

        if self.backend == "hls":
            pass # TODO
        elif self.backend == "chisel":
            buffer_size = self.lcm(self.coarse_in, self.coarse_out)
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
        return pydot.Node(name,label="squeeze3d", shape="polygon",
                sides=4, distortion=distortion, style="filled",
                fillcolor="olive", fontsize=MODULE_3D_FONTSIZE)

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows                       , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols                       , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth                      , "ERROR: invalid depth dimension"
        assert data.shape[3] == self.channels//self.coarse_in   , "ERROR: invalid channel dimension"
        assert data.shape[4] == self.coarse_in                  , "ERROR: invalid coarse dimension"

        out = np.ndarray((
            self.rows,
            self.cols,
            self.depth,
            self.channels//self.coarse_out,
            self.coarse_out),dtype=float)

        out = np.reshape(data,(self.rows,self.cols,self.depth,self.channels))
        out = np.reshape(data,(self.rows,self.cols,self.depth,self.channels//self.coarse_out,self.coarse_out))

        return out


