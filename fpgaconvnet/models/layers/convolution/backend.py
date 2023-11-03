import importlib
import math
from typing import Union, List
from dataclasses import dataclass, field

import pydot
import numpy as np
from dacite import from_dict

from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.models.modules import SlidingWindow, Fork, Conv, Accum, Glue, Bias, ShiftScale, Squeeze, VectorDot

@dataclass(kw_only=True)
class ConvolutionLayerTraitHLS:

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        # create all the modules
        self.modules["sliding_window"] = SlidingWindow(self.rows_in(),
                self.cols_in(), self.channels_in()//self.streams_in(), self.kernel_size,
                self.stride, self.pad_top, self.pad_right, self.pad_bottom, self.pad_left,
                backend="hls", regression_model=self.regression_model)

        self.modules["fork"] = Fork(self.rows_out(),
                self.cols_out(), self.channels_in()//self.streams_in(),
                self.kernel_size, self.coarse_out, backend="hls",
                regression_model=self.regression_model)

        self.modules["Conv"] = Conv(self.rows_out(),
                self.cols_out(), self.channels_in()//self.streams_in(),
                self.filters//self.streams_out(),
                self.fine, self.kernel_size, self.groups//self.coarse_group,
                backend="hls", regression_model=self.regression_model)

        self.modules["accum"] = Accum(self.rows_out(),
                self.cols_out(), self.channels_in()//self.streams_in(),
                self.filters//self.streams_out(),
                self.groups//self.coarse_group,backend="hls",
                regression_model=self.regression_model)

        self.modules["glue"] = Glue(self.rows_out(), self.cols_out(), 1,
                int(self.filters/self.coarse_out), self.coarse_in, self.coarse_out,
                self.coarse_group, backend="hls", regression_model=self.regression_model)

        self.modules["bias"] = Bias(self.rows_out(), self.cols_out(), 1,
                self.filters//self.streams_out(), backend="hls",
                regression_model=self.regression_model)


    def get_fork_parameters(self):
        param = super().get_fork_parameters()
        param["kernel_size"] = self.kernel_size
        return param

    def get_conv_parameters(self):
        return {
            "rows": self.rows_out(),
            "cols": self.cols_out(),
            "channels": self.channels_in()//self.streams_in(),
            "filters": self.filters//self.streams_out(),
            "groups": self.groups//self.coarse_group(),
            "fine": self.fine,
            "data_width": self.input_t.width,
            "weight_width": self.weight_t.width,
            "acc_width": self.acc_t.width,
        }

    def get_accum_parameters(self):
        param = super().get_accum_parameters()
        param["channels"] = self.channels_in()//self.streams_in()
        param["filters"] = self.filters//self.streams_out()
        param["groups"] = self.groups//self.coarse_group()
        return param

    def update_modules(self):

        # iterate over the modules
        for module in self.modules:
            match module:
                case "sliding_window":
                    param = self.get_sliding_window_parameters()
                    for p, v in param.items():
                        setattr(self.modules[module], p, v)
                case "fork":
                    param = self.get_fork_parameters()
                    for p, v in param.items():
                        setattr(self.modules[module], p, v)
                case "conv":
                    param = self.get_conv_parameters()
                    for p, v in param.items():
                        setattr(self.modules[module], p, v)
                case "accum":
                    param = self.get_accum_parameters()
                    for p, v in param.items():
                        setattr(self.modules[module], p, v)
                case "glue":
                    param = self.get_glue_parameters()
                    for p, v in param.items():
                        setattr(self.modules[module], p, v)
                case "bias":
                    param = self.get_bias_parameters()
                    for p, v in param.items():
                        setattr(self.modules[module], p, v)

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
class ConvolutionLayerTraitChisel:

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        self.modules["sliding_window"] = SlidingWindow(self.rows_in(),
                self.cols_in(), self.channels_in()//self.streams_in(), self.kernel_size,
                self.stride, self.pad_top, self.pad_right, self.pad_bottom, self.pad_left,
                backend="chisel", regression_model=self.regression_model)

        self.modules["squeeze"] = Squeeze(self.rows_out(),
                self.cols_out(), self.channels_in()//self.streams_in(),
                np.prod(self.kernel_size), self.fine, backend="chisel",
                regression_model=self.regression_model)

        self.modules["fork"] = Fork(self.rows_out(),
                self.cols_out(), self.channels_in()//self.streams_in(),
                [self.fine, 1], self.coarse_out, backend="chisel",
                regression_model=self.regression_model)

        self.modules["vector_dot"] = VectorDot(self.rows_out(), self.cols_out(),
                (self.channels*np.prod(self.kernel_size))//(self.fine*self.streams_in()),
                self.filters//(self.coarse_out*self.groups), self.fine,
                backend="chisel", regression_model=self.regression_model)

        self.modules["accum"] = Accum(self.rows_out(), self.cols_out(),
                (np.prod(self.kernel_size)*self.channels_in())//(
                    self.fine*self.streams_in()),
                self.filters//(self.streams_out()), 1,
                backend="chisel", regression_model=self.regression_model)

        self.modules["glue"] = Glue(self.rows_out(), self.cols_out(), 1,
                int(self.filters/self.coarse_out), self.coarse_in, self.coarse_out, self.coarse_group,
                backend="chisel", regression_model=self.regression_model) # TODO

        self.modules["bias"] = Bias(self.rows_out(), self.cols_out(), 1, self.filters//self.streams_out(),
                backend="chisel", regression_model=self.regression_model) # TODO

        # update modules
        self.update_modules()

    def get_squeeze_parameters(self):
        return {
            "rows": self.rows_out(),
            "cols": self.cols_out(),
            "channels": self.channels_in()//self.streams_in(),
            "coarse_in": np.prod(self.kernel_size),
            "coarse_out": self.fine,
            "data_width": self.input_t.width
        }

    def get_fork_parameters(self):
        param = super().get_fork_parameters()
        param["kernel_size"] = [self.fine, 1]
        return param

    def get_vector_dot_parameters(self):
        return {
            "rows": self.rows_out(),
            "cols": self.cols_out(),
            "channels": (self.channels*np.prod(self.kernel_size))//(self.fine*self.streams_in()),
            "filters": self.filters//(self.coarse_out*self.groups),
            "data_width": self.input_t.width,
            "weight_width": self.weight_t.width,
            "acc_width": self.acc_t.width,
            "fine": self.fine
        }

    def get_accum_parameters(self):
        param = super().get_accum_parameters()
        param["channels"] = (np.prod(self.kernel_size)*self.channels_in())//(self.fine*self.streams_in())
        param["filters"] = self.filters//self.streams_out()
        param["groups"] = 1
        return param

    def update_modules(self):

        # iterate over the modules
        for module in self.modules:
            match module:
                case "sliding_window":
                    param = self.get_sliding_window_parameters()
                    for p, v in param.items():
                        setattr(self.modules[module], p, v)
                case "squeeze":
                    param = self.get_squeeze_parameters()
                    for p, v in param.items():
                        setattr(self.modules[module], p, v)
                case "fork":
                    param = self.get_fork_parameters()
                    for p, v in param.items():
                        setattr(self.modules[module], p, v)
                case "vector_dot":
                    param = self.get_vector_dot_parameters()
                    for p, v in param.items():
                        setattr(self.modules[module], p, v)
                case "accum":
                    param = self.get_accum_parameters()
                    for p, v in param.items():
                        setattr(self.modules[module], p, v)
                case "glue":
                    param = self.get_glue_parameters()
                    for p, v in param.items():
                        setattr(self.modules[module], p, v)
                case "bias":
                    param = self.get_bias_parameters()
                    for p, v in param.items():
                        setattr(self.modules[module], p, v)

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
            rsc[rsc_type] += sw_rsc[rsc_type]*(self.streams_in()//self.modules['sliding_window'].streams)
            rsc[rsc_type] += squeeze_rsc[rsc_type]*(self.streams_in()//self.modules['squeeze'].streams)
            rsc[rsc_type] += fork_rsc[rsc_type]*(self.streams_in()//self.modules['fork'].streams)
            rsc[rsc_type] += vector_dot_rsc[rsc_type]*((self.streams_in()*self.coarse_out)//self.modules['vector_dot'].streams)
            rsc[rsc_type] += accum_rsc[rsc_type]*((self.streams_in()*self.coarse_out)//self.modules['accum'].streams)
            rsc[rsc_type] += glue_rsc[rsc_type]
            rsc[rsc_type] += bias_rsc[rsc_type]*(self.streams_out()//self.modules['bias'].streams)

        # get the weights resources
        weights_bram, weights_uram = self.get_weight_resources()
        rsc["BRAM"] += weights_bram
        rsc["URAM"] = weights_uram

        # return the resource usage
        return rsc

