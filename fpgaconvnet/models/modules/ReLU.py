"""
.. figure:: ../../../figures/relu_diagram.png
"""

import math
import os
from dataclasses import dataclass, field

import numpy as np
import pydot

from fpgaconvnet.models.modules import Module, MODULE_FONTSIZE

@dataclass
class ReLU(Module):

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
            "DSP"   : 0
        }

    def visualise(self, name):
        return pydot.Node(name,label="relu", shape="box",
                style="filled", fillcolor="dimgrey",
                fontsize=MODULE_FONTSIZE)

    def functional_model(self, data):

        # maximum of 0 and the data
        return np.maximum(data, 0.0)

