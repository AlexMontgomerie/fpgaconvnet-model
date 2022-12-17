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
    backend: str = "chisel"
    regression_model: str = "linear_regression"

    def __post_init__(self):
        # TODO this is a hack for now FIXME
        return
        # get the cache path
        rsc_cache_path = os.path.dirname(__file__) + \
                f"/../../coefficients/{self.regression_model}/{self.backend}"

        # iterate over resource types
        self.rsc_coef = {}
        for rsc_type in self.utilisation_model():
            # load the resource coefficients from the 2D version
            coef_path = os.path.join(rsc_cache_path, f"{self.__class__.__name__.split('3D')[0]}_{rsc_type}.npy".lower())
            self.rsc_coef[rsc_type] = np.load(coef_path)


    def rsc(self, coef=None, model=None):
        # TODO this is a hack for now FIXME
        return {
            "LUT"   : 16,
            "FF"    : 35,
            "BRAM"  : 0,
            "DSP"   : 0
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


