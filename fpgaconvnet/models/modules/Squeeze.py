import math
import os
import sys
from dataclasses import dataclass, field

import pydot
import numpy as np

from fpgaconvnet.models.modules import Module

@dataclass
class Squeeze(Module):
    coarse_in: int
    coarse_out: int

    def __post_init__(self):
        # load the resource model coefficients
        self.rsc_coef["LUT"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/squeeze_lut.npy"))
        self.rsc_coef["FF"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/squeeze_ff.npy"))
        self.rsc_coef["BRAM"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/squeeze_bram.npy"))
        self.rsc_coef["DSP"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/squeeze_dsp.npy"))

    def module_info(self):
        # get the base module fields
        info = Module.module_info(self)
        # add module-specific info fields
        info["coarse_in"] = self.coarse_in
        info["coarse_out"] = self.coarse_out
        # return the info
        return info

    def lcm(a, b):
        return abs(a*b) // math.gcd(a, b)

    def visualise(self, name):
        distortion = 0
        if self.coarse_in > self.coarse_out:
            distortion = self.coarse_in/self.coarse_out
        else:
            distortion = -self.coarse_out/self.coarse_in
        return pydot.Node(name,label="squeeze", shape="polygon",
                sides=4, distortion=distortion, style="filled",
                fillcolor="olive")

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


