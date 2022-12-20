"""
The convolution module computes the dot product
between the feature map windows and the coefficients
of the convolution module. This module has a tunable
degree of parallelism across the kernel dot product,
affecting the throughput and number of ports of the
on-chip weights storage.

.. figure:: ../../../figures/conv_diagram.png
"""

import math
import os
import sys
from typing import Union, List
from dataclasses import dataclass, field

import numpy as np
import pydot

from fpgaconvnet.models.modules import int2bits, Module3D, MODULE_3D_FONTSIZE
from fpgaconvnet.tools.resource_analytical_model import dsp_multiplier_resource_model

@dataclass
class Conv3D(Module3D):
    """
    Conv3D hardware model class.

    Attributes
    ----------
    filters: int
        output channel dimension of the featuremap.
    fine: int

    rows: int
        row dimension of input featuremap
    cols: int
        column dimension of input featuremap
    depth: int
        depth dimension of input featuremap
    channels: int
        channel dimension of input featuremap
    data_width: int
        bitwidth of featuremap pixels (default is 16)
    rsc_coef: list
        list of resource model coefficients. Corresponds
        to `LUT`, `BRAM`, `DSP` and `FF` resources in
        that order.
    """

    filters: int
    fine: int
    # kernel_size: Union[List[int], int]
    kernel_rows: int
    kernel_cols: int
    kernel_depth: int
    groups: int
    weight_width: int = field(default=16, init=False)
    acc_width: int = field(default=16, init=False)
    backend: str = "hls"
    regression_model: str = "linear_regression"

    def __post_init__(self):

        # get the cache path
        rsc_cache_path = os.path.dirname(__file__) + \
                f"/../../coefficients/{self.regression_model}/{self.backend}"

        # iterate over resource types
        self.rsc_coef = {}
        for rsc_type in self.utilisation_model():
            # load the resource coefficients from the 2D version
            coef_path = os.path.join(rsc_cache_path, f"{self.__class__.__name__.split('3D')[0]}_{rsc_type}.npy".lower())
            self.rsc_coef[rsc_type] = np.load(coef_path)

    def channels_out(self):
        return int(self.filters/float(self.groups))

    def rate_in(self):
        return self.fine*self.groups/float(self.kernel_rows*self.kernel_cols*self.kernel_depth*self.filters)

    def rate_out(self):
        return self.fine/float(self.kernel_rows*self.kernel_cols*self.kernel_depth)

    def pipeline_depth(self):
        return self.fine

    def module_info(self):
        # get the base module fields
        info = Module3D.module_info(self)
        # add module-specific info fields
        info["filters"] = self.filters
        # info["kernel_size"] = self.kernel_size
        info["kernel_rows"] = self.kernel_rows
        info["kernel_cols"] = self.kernel_cols
        info["kernel_depth"] = self.kernel_depth
        info["groups"] = self.groups
        info["fine"] = self.fine
        # return the info
        return info

    def utilisation_model(self):
        if self.backend == "hls":
            return {
                "LUT"  : np.array([
                    int2bits(self.filters),
                    int2bits(self.cols*self.rows*self.depth),
                    int2bits(self.channels)
                ]),
                "FF"   : np.array([
                    int2bits(self.filters),
                    int2bits(self.cols*self.rows*self.depth),
                    int2bits(self.channels)
                ]),
                "DSP"  : np.array([0]),
                "BRAM" : np.array([0])
            }
        else:
            raise ValueError(f"{self.backend} backend not supported")

    def rsc(self,coef=None, model=None):
        # use module resource coefficients if none are given
        if coef == None:
            coef = self.rsc_coef
        # get an estimate for the dsp usage
        dot_product_dsp = self.fine * dsp_multiplier_resource_model(self.weight_width, self.data_width)
        # get the linear model estimation
        rsc = Module3D.rsc(self, coef, model)
        # update the dsp usage
        rsc["DSP"] = dot_product_dsp
        # set the BRAM usage to zero
        rsc["BRAM"] = 0
        # return the resource model
        return rsc

    def visualise(self, name):
        return pydot.Node(name,label="conv3d", shape="box",
                height=self.kernel_rows,
                width=self.kernel_cols,
                depth=self.kernel_depth,
                style="filled", fillcolor="gold",
                fontsize=MODULE_3D_FONTSIZE)

    def functional_model(self,data,weights):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth    , "ERROR: invalid depth dimension"
        assert data.shape[3] == self.channels   , "ERROR: invalid channel dimension"
        assert data.shape[4] == self.kernel_rows  , "ERROR: invalid kernel row dimension"
        assert data.shape[5] == self.kernel_cols  , "ERROR: invalid kernel column dimension"
        assert data.shape[6] == self.kernel_depth  , "ERROR: invalid kernel depth dimension"
        # check weight dimensionality
        assert weights.shape[0] == self.channels, "ERROR: invalid channel dimension"
        assert weights.shape[1] == int(self.filters/float(self.groups)) , "ERROR: invalid filter dimension"
        assert weights.shape[2] == self.kernel_rows  , "ERROR: invalid kernel row dimension"
        assert weights.shape[3] == self.kernel_cols  , "ERROR: invalid kernel column dimension"
        assert weights.shape[4] == self.kernel_depth  , "ERROR: invalid kernel depth dimension"

        out = np.zeros((
            self.rows,
            self.cols,
            self.depth,
            self.channels,
            int(self.filters/self.groups)
        ),dtype=float)

        for index,_ in np.ndenumerate(out):
            for k1 in range(self.kernel_rows):
                for k2 in range(self.kernel_cols):
                    for k3 in range(self.kernel_depth):
                        out[index] += data[
                        index[0],index[1],index[2],index[3],k1,k2,k3]*weights[
                        index[2],index[3],index[4],k1,k2,k3]

        return out

