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

from fpgaconvnet.models.modules import Module3D, MODULE_3D_FONTSIZE
from fpgaconvnet.tools.resource_model import dsp_multiplier_resource_model

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
    kernel_size: Union[List[int], int]
    groups: int
    weight_width: int = field(default=16, init=False)
    acc_width: int = field(default=16, init=False)

    def __post_init__(self):
        pass
        # format kernel size as a 2 element list
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size, self.kernel_size]
        elif isinstance(self.kernel_size, list):
            assert len(self.kernel_size) == 2, "Must specify two kernel dimensions"
        else:
            raise TypeError

        # load the resource model coefficients
        self.rsc_coef["LUT"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/conv_lut.npy"))
        self.rsc_coef["FF"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/conv_ff.npy"))
        self.rsc_coef["BRAM"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/conv_bram.npy"))
        self.rsc_coef["DSP"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/conv_dsp.npy"))

    def utilisation_model(self):
        pass
        return {
            "LUT"  : np.array([math.log(self.filters,2),math.log(self.cols*self.rows,2),math.log(self.channels,2)]),
            "FF"   : np.array([math.log(self.filters,2),math.log(self.cols*self.rows,2),math.log(self.channels,2)]),
            "DSP"  : np.array([1]),
            "BRAM" : np.array([1])
        }

    def channels_out(self):
        pass
        return int(self.filters/float(self.groups))

    def rate_in(self):
        pass
        return self.fine*self.groups/float(self.kernel_size[0]*self.kernel_size[1]*self.filters)

    def rate_out(self):
        pass
        return self.fine/float(self.kernel_size[0]*self.kernel_size[1])

    def pipeline_depth(self):
        pass
        return self.fine

    def module_info(self):
        pass
        # get the base module fields
        info = Module3D.module_info(self)
        # add module-specific info fields
        info["filters"] = self.filters
        info["kernel_size"] = self.kernel_size
        info["groups"] = self.groups
        info["fine"] = self.fine
        # return the info
        return info

    def rsc(self,coef=None):
        pass
        # use module resource coefficients if none are given
        if coef == None:
            coef = self.rsc_coef
        # get an estimate for the dsp usage
        dot_product_dsp = self.fine * dsp_multiplier_resource_model(self.weight_width, self.data_width)
        # get the linear model estimation
        rsc = Module3D.rsc(self, coef)
        # update the dsp usage
        rsc["DSP"] = dot_product_dsp
        # set the BRAM usage to zero
        rsc["BRAM"] = 0
        # return the resource model
        return rsc

    def visualise(self, name):
        pass
        return pydot.Node(name,label="conv", shape="box",
                height=self.kernel_size[0],
                width=self.kernel_size[1],
                style="filled", fillcolor="gold",
                fontsize=MODULE_3D_FONTSIZE)

    def functional_model(self,data,weights):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth    , "ERROR: invalid depth dimension"
        assert data.shape[3] == self.channels, "ERROR: invalid channel dimension"
        assert data.shape[4] == self.kernel_size[0]  , "ERROR: invalid kernel row dimension"
        assert data.shape[5] == self.kernel_size[1]  , "ERROR: invalid kernel column dimension"
        assert data.shape[6] == self.kernel_size[2]  , "ERROR: invalid kernel depth dimension"
        # check weight dimensionality
        assert weights.shape[0] == self.channels, "ERROR: invalid channel dimension"
        assert weights.shape[1] == int(self.filters/float(self.groups)) , "ERROR: invalid filter dimension"
        assert weights.shape[2] == self.kernel_size[0]  , "ERROR: invalid kernel row dimension"
        assert weights.shape[3] == self.kernel_size[1]  , "ERROR: invalid kernel column dimension"
        assert weights.shape[4] == self.kernel_size[2]  , "ERROR: invalid kernel depth dimension"

        out = np.zeros((
            self.rows,
            self.cols,
            self.depth,
            self.channels,
            int(self.filters/self.groups)
        ),dtype=float)

        for index,_ in np.ndenumerate(out):
            for k1 in range(self.kernel_size[0]):
                for k2 in range(self.kernel_size[1]):
                    for k3 in range(self.kernel_size[2]):
                        out[index] += data[
                        index[0],index[1],index[2],index[3],k1,k2,k3]*weights[
                        index[2],index[3],index[4],k1,k2,k3]

        return out

