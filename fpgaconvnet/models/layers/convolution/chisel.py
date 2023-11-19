import importlib
import math
from typing import Union, ClassVar
from dataclasses import dataclass, field
from collections import OrderedDict

import pydot
import numpy as np
from dacite import from_dict

from fpgaconvnet.models.layers.convolution import ConvolutionLayer2DBase
from fpgaconvnet.models.modules import ModuleBase
from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY
from fpgaconvnet.models.layers.utils import get_factors

@dataclass(kw_only=True)
class ConvolutionLayerChisel(ConvolutionLayer2DBase):

    backend: ClassVar[BACKEND] = BACKEND.CHISEL
    register: ClassVar[bool] = True

    @property
    def module_lookup(self) -> dict:
        return OrderedDict({
            "pad": self.get_pad_parameters,
            "sliding_window": self.get_sliding_window_parameters,
            "squeeze": self.get_squeeze_parameters,
            "fork": self.get_fork_parameters,
            "vector_dot": self.get_vector_dot_parameters,
            "accum": self.get_accum_parameters,
            "glue": self.get_glue_parameters,
            "bias": self.get_bias_parameters
        })

    def get_pad_parameters(self):

        return {
            "repetitions": 1,
            "streams": self.streams_in(),
            "rows": self.rows,
            "cols": self.cols,
            "channels": self.channels,
            "pad_top": self.pad_top,
            "pad_right": self.pad_right,
            "pad_bottom": self.pad_bottom,
            "pad_left": self.pad_left,
            "data_t": self.input_t,
        }

    def get_sliding_window_parameters(self):

        # get the dimensions from the pad module
        rows, cols, channels = self.modules["pad"].output_iter_space[0]

        return {
            "repetitions": 1,
            "streams": self.streams_in(),
            "rows": rows,
            "cols": cols,
            "channels": channels//self.streams_in(),
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "data_t": self.input_t,
        }

    def get_squeeze_parameters(self):

        return {
            "repetitions": self.rows_out()*self.cols_out()*self.channels//self.streams_in(),
            "streams": self.streams_in(),
            "coarse_in": int(np.prod(self.kernel_size)),
            "coarse_out": self.fine,
            "data_t": self.input_t,
        }

    def get_fork_parameters(self):

        return {
            "repetitions": self.rows_out()*self.cols_out()*self.channels//self.streams_in(),
            "streams": self.streams_in(),
            "fine": self.fine,
            "coarse": self.coarse_out,
            "data_t": self.input_t,
        }

    def get_vector_dot_parameters(self):

        repetitions = self.rows_out() * self.cols_out() * \
                self.channels//self.streams_in() * \
                int(np.prod(self.kernel_size))//self.fine

        return {
            "repetitions": repetitions,
            "streams": self.coarse_in*self.coarse_out*self.coarse_group,
            "filters": self.filters//self.streams_out(),
            "fine": self.fine,
            "data_t": self.input_t,
            "weight_t": self.weight_t,
            "acc_t": self.acc_t,
        }

    def get_accum_parameters(self):

        channels = self.channels//self.streams_in() * \
                int(np.prod(self.kernel_size))//self.fine

        return {
            "repetitions": self.rows_out() * self.cols_out(),
            "streams": self.coarse_in*self.coarse_out*self.coarse_group,
            "channels": channels,
            "filters": self.filters//self.streams_out(),
            "data_t": self.acc_t,
        }

    def get_glue_parameters(self):

        return {
            "repetitions": self.rows_out()*self.cols_out()*self.channels_out(),
            "coarse": self.coarse_in,
            "streams": self.streams_out(),
            "data_t": self.acc_t,
        }

    def get_bias_parameters(self):

        return {
            "repetitions": self.rows_out()*self.cols_out(),
            "streams": self.streams_out(),
            "channels": self.filters,
            "data_t": self.output_t,
        }


    def get_fine_feasible(self):
        return get_factors(np.prod(self.kernel_size))

    def resource(self):

        # get module resource models
        sw_rsc          = self.modules['sliding_window'].rsc()
        squeeze_rsc     = self.modules['squeeze'].rsc()
        fork_rsc        = self.modules['fork'].rsc()
        vector_dot_rsc  = self.modules['vector_dot'].rsc()
        accum_rsc       = self.modules['accum'].rsc()
        glue_rsc        = self.modules['glue'].rsc()
        bias_rsc        = self.modules['bias'].rsc()

        # remove redundant modules
        if self.pointwise:
            sw_rsc = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}

        # accumulate resource usage based on coarse factors
        rsc = { rsc_type: 0 for rsc_type in ["LUT", "FF", "DSP", "BRAM"] }

        # iterate over the resource types
        for rsc_type in ["LUT", "FF", "DSP", "BRAM"]:

            # add each of the modules
            rsc[rsc_type] += sw_rsc[rsc_type]
            rsc[rsc_type] += squeeze_rsc[rsc_type]
            rsc[rsc_type] += fork_rsc[rsc_type]
            rsc[rsc_type] += vector_dot_rsc[rsc_type]
            rsc[rsc_type] += accum_rsc[rsc_type]
            rsc[rsc_type] += glue_rsc[rsc_type]
            rsc[rsc_type] += bias_rsc[rsc_type]

        # get the weights resources
        weights_bram, weights_uram = self.get_weight_resources()
        rsc["BRAM"] += weights_bram
        rsc["URAM"] = weights_uram

        # return the resource usage
        return rsc

