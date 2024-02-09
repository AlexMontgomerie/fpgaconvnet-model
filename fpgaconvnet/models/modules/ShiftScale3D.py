"""
.. figure:: ../../../figures/shift_scale_diagram.png
"""
import numpy as np
import math
import os
from dataclasses import dataclass, field

from fpgaconvnet.models.modules import Module3D

@dataclass
class ShiftScale3D(Module3D):
    filters: int
    biases_width: int = field(default=16, init=False)
    backend: str = "chisel"
    regression_model: str = "linear_regression"
    streams: int = 1

    def __post_init__(self):
        pass

    def rsc(self):
        return {
          "LUT"  : 0, 
          "BRAM" : 0,
          "DSP"  : 1*self.streams,
          "FF"   : 0 
        }

    def functional_model(self, data, scale, shift):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth , "ERROR: invalid filter dimension"
        assert data.shape[3] == self.channels, "ERROR: invalid channel dimension"

        out = np.ndarray((
            self.rows,
            self.cols,
            self.depth,
            self.channels),dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = scale[index[3]] * ( data[index] + shift[index[3]] )

        return out


