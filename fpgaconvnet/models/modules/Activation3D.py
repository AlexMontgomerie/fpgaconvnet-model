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
class Activation3D(Module3D):
    activation_type: str
    backend: str = "chisel"
    regression_model: str = "linear_regression"

    def __post_init__(self):
        # TODO this is a hack for now FIXME
        return
        # self.__class__.__name__ = f"{self.activation_type.capitalize()}3D"
        # # get the cache path
        # rsc_cache_path = os.path.dirname(__file__) + \
        #         f"/../../coefficients/{self.regression_model}/{self.backend}"

        # # iterate over resource types
        # self.rsc_coef = {}
        # for rsc_type in self.utilisation_model():
        #     # load the resource coefficients from the 2D version
        #     coef_path = os.path.join(rsc_cache_path, f"{self.__class__.__name__.split('3D')[0]}_{rsc_type}.npy".lower())
        #     self.rsc_coef[rsc_type] = np.load(coef_path)

    def rsc(self, coef=None, model=None):
        # TODO this is a hack for now FIXME
        if self.activation_type == "relu":
            return {
                "LUT"   : 16,
                "FF"    : 35,
                "BRAM"  : 0,
                "DSP"   : 0
            }
        elif self.activation_type == "sigmoid":
            return {
                "LUT"   : 16,
                "FF"    : 35,
                "BRAM"  : 0,
                "DSP"   : 3
            }
        elif self.activation_type == "silu":
            return {
                "LUT"   : 16,
                "FF"    : 35,
                "BRAM"  : 0,
                "DSP"   : 4
            }
        else:
            raise NotImplementedError(f"Function not implemented for activation type {self.activation_type}")

    def visualise(self, name):
        return pydot.Node(name,label=f"{self.activation_type}3d", shape="box",
                style="filled", fillcolor="dimgrey",
                fontsize=MODULE_3D_FONTSIZE)

    def functional_model(self, data):
        import torch

        # check input dimensionality
        assert data.shape[0] == self.rows     , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols     , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth    , "ERROR: invalid depth dimension"
        assert data.shape[3] == self.channels , "ERROR: invalid channel dimension"

        if self.activation_type == "relu":
            activation_op = torch.nn.ReLU()
        elif self.activation_type == "sigmoid":
            activation_op = torch.nn.Sigmoid()
        elif self.activation_type == "silu":
            activation_op = torch.nn.SiLU()
        else:
            raise ValueError(f"ERROR: invalid activation type {self.activation_type}")

        out = activation_op(torch.from_numpy(data)).numpy()

        return out
