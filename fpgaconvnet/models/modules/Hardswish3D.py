"""
.. figure:: ../../../figures/hardswish_diagram.png
"""

import math
import os
from dataclasses import dataclass, field

import numpy as np
import pydot

from fpgaconvnet.models.modules import Module3D, MODULE_3D_FONTSIZE

@dataclass
class Hardswish3D(Module3D):
    backend: str = "chisel"
    regression_model: str = "linear_regression"

    def __post_init__(self):
        pass

    def rsc(self, coef=None, model=None):
        """
        Returns
        -------
        dict
            estimated resource usage of the module. Uses the
            resource coefficients for the estimate.
        """
        return {
            "LUT"   : 16,
            "FF"    : 35,
            "BRAM"  : 0,
            "DSP"   : 5
        }

    def visualise(self, name):
        return pydot.Node(name,label="hardswish3d", shape="box",
                style="filled", fillcolor="dimgrey",
                fontsize=MODULE_3D_FONTSIZE)

    @staticmethod
    def hardswish(x):
        return x * max(0, min(1, (x + 3) / 6))

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth   , "ERROR: invalid depth dimension"
        assert data.shape[3] == self.channels, "ERROR: invalid channel dimension"

        out = np.ndarray((
            self.rows,
            self.cols,
            self.depth,
            self.channels),dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = self.hardswish(data[index])

        return out


