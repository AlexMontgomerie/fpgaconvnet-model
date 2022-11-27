"""
.. figure:: ../../../figures/relu_diagram.png
"""

import math
import os
from dataclasses import dataclass, field

import numpy as np
import pydot

from fpgaconvnet.models.modules import Module3D, MODULE_3D_FONTSIZE

@dataclass
class ReLU3D(Module3D):

    def __post_init__(self):
        # load the resource model coefficients
        # TODO: Update resource model coefficients FIXME
        self.rsc_coef["LUT"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/relu3d_lut.npy"))
        self.rsc_coef["FF"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/relu3d_ff.npy"))
        self.rsc_coef["BRAM"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/relu3d_bram.npy"))
        self.rsc_coef["DSP"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/relu3d_dsp.npy"))

    def utilisation_model(self):
        # TODO: Update utilisation model FIXME
        return {
            "LUT"  : np.array([self.data_width, math.ceil(math.log(self.channels*self.rows*self.cols*self.depth,2))]),
            "FF"   : np.array([self.data_width, math.ceil(math.log(self.channels*self.rows*self.cols*self.depth,2))]),
            "DSP"  : np.array([1]),
            "BRAM" : np.array([1])
        }

    def visualise(self, name):
        return pydot.Node(name,label="relu3d", shape="box",
                style="filled", fillcolor="dimgrey",
                fontsize=MODULE_3D_FONTSIZE)

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows     , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols     , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth    , "ERROR: invalid depth dimension"
        assert data.shape[3] == self.channels , "ERROR: invalid channel dimension"

        out = np.ndarray((
            self.rows,
            self.cols,
            self.depth,
            self.channels),dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = max(data[index],0.0)

        return out


