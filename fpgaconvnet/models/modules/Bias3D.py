"""
The bias module implements the addition of the
bias term for the convolution (and inner product)
layers when applicable. Each of element of the bias
vector is added to its corresponding output feature
map.

Figure pending.
"""

import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pydot

from fpgaconvnet.models.modules import Module3D, MODULE_3D_FONTSIZE
#from fpgaconvnet.tools.resource_analytical_model import bram_memory_resource_model
from fpgaconvnet.tools.resource_analytical_model import dsp_multiplier_resource_model

@dataclass
class Bias3D(Module3D):
    filters: int
    biases_width: int = field(default=16, init=False)

    def __post_init__(self):
        # load the resource model coefficients
        # TODO: Update resource model coefficients FIXME
        self.rsc_coef["LUT"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/accum3d_lut.npy"))
        self.rsc_coef["FF"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/accum3d_ff.npy"))
        self.rsc_coef["BRAM"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/accum3d_bram.npy"))
        self.rsc_coef["DSP"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/accum3d_dsp.npy"))



    def utilisation_model(self):
        # TODO: Update utilisation model FIXME
        return {
            "LUT"   : np.array([self.filters,1,self.data_width,self.cols,self.rows,self.depth,1]),
            "FF"    : np.array([self.filters,1,self.data_width,self.cols,self.rows,self.depth,1]),
            "DSP"   : np.array([self.filters,1,self.data_width,self.cols,self.rows,self.depth,1]),
            "BRAM"  : np.array([self.filters,1,self.data_width,self.cols,self.rows,self.depth,1]),
        }

    def channels_in(self):
        return self.filters

    def channels_out(self):
        return self.filters

    def module_info(self):#TODO
        # get the base module fields
        info = Module3D.module_info(self)
        # add module-specific info fields

        info['filters'] = self.filters

        # return the info
        return info

    def rsc(self,coef=None):#TODO replace conv version of func
        # use module resource coefficients if none are given
        if coef == None:
            coef = self.rsc_coef
        # get an estimate for the dsp usage
        dot_product_dsp = dsp_multiplier_resource_model(self.biases_width, self.data_width)
        # get the linear model estimation
        rsc = Module3D.rsc(self, coef)
        # update the dsp usage
        rsc["DSP"] = dot_product_dsp
        # set the BRAM usage to zero
        rsc["BRAM"] = 0
        # return the resource model
        return rsc

    def visualise(self, name):
        return pydot.Node(name,label="bias3d", shape="box",
                style="filled", fillcolor="chartreuse",
                fontsize=MODULE_3D_FONTSIZE)


    def functional_model(self,data,biases):
        # check input dimensionality
        assert data.shape[0] == self.rows                   , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols                   , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth                  , "ERROR: invalid depth dimension"
        assert data.shape[3] == self.filters                , "ERROR: invalid filter dimension"
        # check bias dimensionality
        assert biases.shape[0] == self.filters              , "ERROR: invalid filter dimension"

        out = np.zeros((
            self.rows,
            self.cols,
            self.depth,
            self.filters,
            ), dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = data[index] + biases[index[3]]

        # sanity check because numpy indexing confuses me
        for f_i in range(self.filters):
            # create copy of input and output filter
            cf = np.empty_like(data[:,:,:,0])
            cfo = np.empty_like(data[:,:,:,0])
            # set values of input and output
            cf[:] = data[:,:,:,f_i]
            cfo[:] = out[:,:,:,f_i]
            # subtraction should give bias
            v = cfo - cf
            for _,val in np.ndenumerate(v):
                # check each filter result has been added correctly to the bias
                assert np.allclose(biases[f_i],val,
                        rtol=1.e-8,atol=1.e-8), "ERROR: the biases don't match!"
        return out
