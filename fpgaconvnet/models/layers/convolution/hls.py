import importlib
import math
from typing import Union, ClassVar
from dataclasses import dataclass, field
from collections import OrderedDict

import pydot
import numpy as np
from dacite import from_dict

from fpgaconvnet.models.layers.convolution import ConvolutionLayer2DBase, ConvolutionLayer3DBase
from fpgaconvnet.models.modules import ModuleBase
from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY
from fpgaconvnet.models.layers.utils import get_factors

@dataclass(kw_only=True)
class ConvolutionLayerHLS(ConvolutionLayer2DBase):

    backend: ClassVar[BACKEND] = BACKEND.HLS
    register: ClassVar[bool] = True

    @property
    def module_lookup(self) -> dict:
        return OrderedDict({
            "sliding_window": self.get_sliding_window_parameters,
            "fork": self.get_fork_parameters,
            "conv": self.get_conv_parameters,
            "accum": self.get_accum_parameters,
            "glue": self.get_glue_parameters,
            "bias": self.get_bias_parameters
        })

    def get_sliding_window_parameters(self):

        return {
            "rows": self.rows,
            "cols": self.cols,
            "channels": self.channels//self.streams_in(),
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "pad": self.pad,
            "data_t": self.input_t,
        }

    def get_fork_parameters(self):

        # check that the sliding_window is intialised
        assert "sliding_window" in self.modules.keys()

        return {
            "rows": self.rows_out(),
            "cols": self.cols_out(),
            "channels": self.channels//self.streams_in(),
            "kernel_size": self.kernel_size,
            "coarse": self.coarse_out,
            "data_t": self.input_t,
        }

    def get_conv_parameters(self):

        # check that the fork is intialised
        assert "fork" in self.modules.keys()

        return {
            "rows": self.rows_out(),
            "cols": self.cols_out(),
            "channels": self.channels//self.streams_in(),
            "filters": self.filters//self.streams_out(),
            "groups": self.groups,
            "kernel_size": self.kernel_size,
            "fine": self.fine,
            "data_t": self.input_t,
            "weight_t": self.weight_t,
            "acc_t": self.acc_t,
        }

    def get_accum_parameters(self):

        # check that the conv is intialised
        assert "conv" in self.modules.keys()

        return {
            "rows": self.rows_out(),
            "cols": self.cols_out(),
            "channels": self.channels//self.streams_in(),
            "filters": self.filters//self.streams_out(),
            "groups": self.groups,
            "data_t": self.acc_t,
        }

    def get_glue_parameters(self):

        # check that the accum is intialised
        assert "accum" in self.modules.keys()

        return {
            "rows": self.rows_out(),
            "cols": self.cols_out(),
            "channels": self.channels//self.streams_in(),
            "filters": self.filters//self.streams_out(),
            "coarse_in": self.coarse_in,
            "coarse_out": self.coarse_out,
            "coarse_group": self.coarse_group,
            "data_t": self.acc_t,
        }

    def get_bias_parameters(self):

        # check that the glue is intialised
        assert "glue" in self.modules.keys()

        return {
            "rows": self.rows_out(),
            "cols": self.cols_out(),
            "channels": self.channels//self.streams_in(),
            "filters": self.filters//self.streams_out(),
            "data_t": self.acc_t,
        }

    def get_fine_feasible(self):
        return [ 1, self.kernel_size[0], self.kernel_size[0]*self.kernel_size[1] ] # TODO: extend to 3D case

    def resource(self):

        # get module resource models
        sw_rsc      = self.modules['sliding_window'].rsc()
        fork_rsc    = self.modules['fork'].rsc()
        conv_rsc    = self.modules['conv'].rsc()
        accum_rsc   = self.modules['accum'].rsc()
        glue_rsc    = self.modules['glue'].rsc()
        bias_rsc    = self.modules['bias'].rsc()

        # remove redundant modules
        if self.pointwise:
            sw_rsc = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}

        # accumulate resource usage based on coarse factors
        rsc = { rsc_type: 0 for rsc_type in ["LUT", "FF", "DSP", "BRAM"] }

        # iterate over the resource types
        for rsc_type in ["LUT", "FF", "DSP", "BRAM"]:

            # add each of the modules
            rsc[rsc_type] += sw_rsc[rsc_type]*self.streams_in()
            rsc[rsc_type] += fork_rsc[rsc_type]*self.streams_in()
            rsc[rsc_type] += conv_rsc[rsc_type]*self.streams_in()*self.coarse_out
            rsc[rsc_type] += accum_rsc[rsc_type]*self.streams_in()*self.coarse_out
            rsc[rsc_type] += glue_rsc[rsc_type]
            rsc[rsc_type] += bias_rsc[rsc_type]*self.streams_out()

        # get the weights resources
        weights_bram, weights_uram = self.get_weight_resources()
        rsc["BRAM"] += weights_bram
        rsc["URAM"] = weights_uram

        # return the resource usage
        return rsc

@dataclass(kw_only=True)
class ConvolutionLayer3DHLS(ConvolutionLayerHLS, ConvolutionLayer3DBase):

    register: ClassVar[bool] = True

    def get_sliding_window_parameters(self):
        param = super().get_sliding_window_parameters()
        param["depth"] = self.depth

    def get_fork_parameters(self):
        param = super().get_fork_parameters()
        param["depth"] = self.depth_out()

    def get_conv_parameters(self):
        param = super().get_conv_parameters()
        param["depth"] = self.depth_out()

    def get_accum_parameters(self):
        param = super().get_accum_parameters()
        param["depth"] = self.depth_out()

    def get_glue_parameters(self):
        param = super().get_glue_parameters()
        param["depth"] = self.depth_out()

    def get_bias_parameters(self):
        param = super().get_bias_parameters()
        param["depth"] = self.depth_out()

