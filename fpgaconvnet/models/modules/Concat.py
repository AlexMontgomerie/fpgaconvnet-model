"""
"""

import math
import os
import sys
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pydot

from fpgaconvnet.models.modules import Module

@dataclass
class Concat(Module):
    channels: List[int]
    ports_in: int
    biases_width: int = field(default=16, init=False)

    def __post_init__(self):
        # load the resource model coefficients
        #TODO add model coefs FOR BIAS - currently using conv to approx.
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

    def channels_in(self, port_index=0):
        return self.channels[port_index]

    def channels_out(self, port_index=0):
        return sum(self.channels)

    def rate_in(self, port_index=0):
        return self.channels_in(port_index)/float(sum(self.channels))

    def module_info(self):
        return {
            'type'      : self.__class__.__name__.upper(),
            'rows'      : self.rows_in(),
            'cols'      : self.cols_in(),
            'channels'  : self.channels,
            'ports_in'      : self.ports_in,
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out()
        }

    def rsc(self,coef=None):
        if coef == None:
            coef = self.rsc_coef
        return {
          "LUT"  : 0,
          "BRAM" : 0,
          "DSP"  : 0,
          "FF"   : 0,
        }

    def functional_model(self, data):
        # check input dimensionality
        assert len(data) == self.ports_in , "ERROR: invalid row dimension"
        for i in range(self.ports_in):
            assert data[i].shape[0] == self.rows       , "ERROR: invalid column dimension"
            assert data[i].shape[1] == self.cols       , "ERROR: invalid column dimension"
            assert data[i].shape[2] == self.channels[i], "ERROR: invalid channel dimension"

        out = np.ndarray((
            self.rows,
            self.cols,
            sum(self.channels)),dtype=float)

        channel_offset = 0
        for i in range(self.ports_in):
            for index,_ in np.ndenumerate(data[i]):
                out[index[0],index[1],channel_offset+index[2]] = data[i][index]
            channel_offset += self.channels[i]
        return out


