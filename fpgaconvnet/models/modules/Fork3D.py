"""
The Fork module provides functionality for
parallelism within layers. By duplicating the
streams, it can be used for exploiting
parallelism across filters in the Convolution
layers.

.. figure:: ../../../figures/fork_diagram.png
"""

import math
import os
import sys
from typing import Union, List
from dataclasses import dataclass, field

import numpy as np
import pydot

from fpgaconvnet.models.modules import Module3D, MODULE_3D_FONTSIZE

@dataclass
class Fork3D(Module3D):
    # kernel_size: Union[List[int], int]
    kernel_rows: int
    kernel_cols: int
    kernel_depth: int
    coarse: int

    def __post_init__(self):
        # load the resource model coefficients
        # TODO: Update resource model coefficients FIXME
        self.rsc_coef["LUT"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/fork3d_lut.npy"))
        self.rsc_coef["FF"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/fork3d_ff.npy"))
        self.rsc_coef["BRAM"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/fork3d_bram.npy"))
        self.rsc_coef["DSP"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/fork3d_dsp.npy"))

    def utilisation_model(self):
        # TODO: Update utilisation model FIXME
        return {
            "LUT"  : np.array([math.ceil(math.log(self.channels*self.rows*self.cols*self.depth,2))]),
            "FF"   : np.array([math.ceil(math.log(self.channels*self.rows*self.cols*self.depth,2))]),
            "DSP"  : np.array([1]),
            "BRAM" : np.array([1]),
        }

    def module_info(self):
        # get the base module fields
        info = Module3D.module_info(self)
        # add module-specific info fields
        info["coarse"] = self.coarse
        # info["kernel_size"] = self.kernel_size
        info["kernel_rows"] = self.kernel_rows
        info["kernel_cols"] = self.kernel_cols
        info["kernel_depth"] = self.kernel_depth
        # return the info
        return info

    def visualise(self, name):
        return pydot.Node(name,label="fork3d", shape="box",
                style="filled", fillcolor="azure",
                fontsize=MODULE_3D_FONTSIZE)

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth    , "ERROR: invalid depth dimension"
        assert data.shape[3] == self.channels, "ERROR: invalid channel dimension"
        assert data.shape[4] == self.kernel_rows  , "ERROR: invalid kernel row dimension"
        assert data.shape[5] == self.kernel_cols  , "ERROR: invalid kernel column dimension"
        assert data.shape[6] == self.kernel_depth  , "ERROR: invalid kernel depth dimension"

        out = np.ndarray((
            self.rows,
            self.cols,
            self.depth,
            self.channels,
            self.coarse,
            self.kernel_rows,
            self.kernel_cols,
            self.kernel_depth),dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = data[
              index[0],
              index[1],
              index[2],
              index[3],
              index[5],
              index[6],
              index[7]]

        return out

