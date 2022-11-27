"""
.. figure:: ../../../figures/relu_diagram.png
"""

import math
import os
from dataclasses import dataclass, field

import torch
import numpy as np
import pydot

from fpgaconvnet.models.modules import Module3D, MODULE_3D_FONTSIZE

@dataclass
class Activation3D(Module3D):
    activation_type: str

    def __post_init__(self):
        # load the resource model coefficients
        # TODO: Update resource model coefficients FIXME
        if self.activation_type == "relu":
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
        elif self.activation_type == "sigmoid":
            self.rsc_coef["LUT"] = np.load(
                    os.path.join(os.path.dirname(__file__),
                    "../../coefficients/sigmoid3d_lut.npy"))
            self.rsc_coef["FF"] = np.load(
                    os.path.join(os.path.dirname(__file__),
                    "../../coefficients/sigmoid3d_ff.npy"))
            self.rsc_coef["BRAM"] = np.load(
                    os.path.join(os.path.dirname(__file__),
                    "../../coefficients/sigmoid3d_bram.npy"))
            self.rsc_coef["DSP"] = np.load(
                    os.path.join(os.path.dirname(__file__),
                    "../../coefficients/sigmoid3d_dsp.npy"))
        elif self.activation_type == "silu":
            self.rsc_coef["LUT"] = np.load(
                    os.path.join(os.path.dirname(__file__),
                    "../../coefficients/silu3d_lut.npy"))
            self.rsc_coef["FF"] = np.load(
                    os.path.join(os.path.dirname(__file__),
                    "../../coefficients/silu3d_ff.npy"))
            self.rsc_coef["BRAM"] = np.load(
                    os.path.join(os.path.dirname(__file__),
                    "../../coefficients/silu3d_bram.npy"))
            self.rsc_coef["DSP"] = np.load(
                    os.path.join(os.path.dirname(__file__),
                    "../../coefficients/silu3d_dsp.npy"))
        else:
            raise ValueError(f"ERROR: invalid activation type {self.activation_type}")

    def utilisation_model(self):
        # TODO: Update utilisation model FIXME
        if self.activation_type == "relu":
            return {
                "LUT"  : np.array([self.data_width, math.ceil(math.log(self.channels*self.rows*self.cols*self.depth,2))]),
                "FF"   : np.array([self.data_width, math.ceil(math.log(self.channels*self.rows*self.cols*self.depth,2))]),
                "DSP"  : np.array([1]),
                "BRAM" : np.array([1])
            }
        elif self.activation_type == "sigmoid":
            return {
                "LUT"  : np.array([self.data_width, math.ceil(math.log(self.channels*self.rows*self.cols*self.depth,2))]),
                "FF"   : np.array([self.data_width, math.ceil(math.log(self.channels*self.rows*self.cols*self.depth,2))]),
                "DSP"  : np.array([3]),
                "BRAM" : np.array([1])
            }
        elif self.activation_type == "silu":
            return {
                "LUT"  : np.array([self.data_width, math.ceil(math.log(self.channels*self.rows*self.cols*self.depth,2))]),
                "FF"   : np.array([self.data_width, math.ceil(math.log(self.channels*self.rows*self.cols*self.depth,2))]),
                "DSP"  : np.array([4]),
                "BRAM" : np.array([1])
            }
        else:
            raise ValueError(f"ERROR: invalid activation type {self.activation_type}")

    def visualise(self, name):
        return pydot.Node(name,label=f"{self.activation_type}3d", shape="box",
                style="filled", fillcolor="dimgrey",
                fontsize=MODULE_3D_FONTSIZE)

    def functional_model(self, data):
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