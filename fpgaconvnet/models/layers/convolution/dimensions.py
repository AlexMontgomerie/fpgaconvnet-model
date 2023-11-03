import importlib
import math
from typing import Union, List
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field

import pydot
import numpy as np
from dacite import from_dict

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

from fpgaconvnet.models.layers import LayerBaseMeta, Layer, Layer3D
from fpgaconvnet.models.modules import *

from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY, SPARSITY

from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

@dataclass(kw_only=True)
class ConvolutionLayerTrait2D:
    kernel_rows: int = 1
    kernel_cols: int = 1
    stride_rows: int = 1
    stride_cols: int = 1
    pad_top: int = 0
    pad_right: int = 0
    pad_bottom: int = 0
    pad_left: int = 0

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

    def get_sliding_window_parameters(self):
        return {
            "rows"      : self.rows_in(),
            "cols"      : self.cols_in(),
            "channels"  : self.channels_in()//self.streams_in(),
            "data_width": self.input_t.width,
        }

    def get_fork_parameters(self):
        return {
            "rows"      : self.rows_out(),
            "cols"      : self.cols_out(),
            "channels"  : self.channels_in()//self.streams_in(),
            "coarse"    : self.coarse_out,
            "data_width": self.input_t.width,
        }

    def get_accum_parameters(self):
        return {
            "rows"      : self.rows_out(),
            "cols"      : self.cols_out(),
            "data_width": self.acc_t.width,
        }

    def get_glue_parameters(self):
        return {
            "rows"      : self.rows_out(),
            "cols"      : self.cols_out(),
            "filters"   : self.filters//self.coarse_group,
            "coarse_in" : self.coarse_in,
            "coarse_out": self.coarse_out,
            "coarse_group": self.coarse_group,
            "data_width": self.acc_t.width,
        }

    def get_bias_parameters(self):
        return {
            "rows"      : self.rows_out(),
            "cols"      : self.cols_out(),
            "filters"   : self.filters//self.streams_out(),
            "data_width": self.acc_t.width,
        }

    @property
    def kernel_size(self) -> List[int]:
        return [ self.kernel_rows, self.kernel_cols ]

    @property
    def stride(self) -> List[int]:
        return [ self.stride_rows, self.stride_cols ]

    @property
    def pad(self) -> List[int]:
        return [
            self.pad_top,
            self.pad_left,
            self.pad_bottom,
            self.pad_right,
        ]

    @kernel_size.setter
    def kernel_size(self, val: List[int]) -> None:
        assert(len(val) == 2, "kernel size must be a list of two integers")
        self.kernel_rows = val[0]
        self.kernel_cols = val[1]

    @stride.setter
    def stride(self, val: List[int]) -> None:
        assert(len(val) == 2, "stride must be a list of two integers")
        self.stride_rows = val[0]
        self.stride_cols = val[1]

    @pad.setter
    def pad(self, val: List[int]) -> None:
        assert(len(val) == 4, "pad must be a list of four integers")
        self.pad_top    = val[0]
        self.pad_right  = val[3]
        self.pad_bottom = val[2]
        self.pad_left   = val[1]

    def rows_out(self) -> int:
        return self.modules["sliding_window"].rows_out()

    def cols_out(self) -> int:
        return self.modules["sliding_window"].cols_out()

    def pipeline_depth(self):
        # pipeline depth of the sliding window minus the total words in the pipeline from padding
        # plus the words needed to fill the accum buffer
        return (self.kernel_rows-1)*(self.cols+self.pad_left+self.pad_right)*self.channels//self.streams_in() + \
                (self.kernel_cols-1)*self.channels//self.streams_in() - \
                ( self.pad_top * self.cols * self.channels//self.streams_in() + \
                (self.pad_left+self.pad_right)*self.channels//self.streams_in() ) + \
                self.channels//self.streams_in()

    def functional_model(self,data,weights,bias,batch_size=1):
        import torch

        assert data.shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels_in(), "ERROR (data): invalid channel dimension"

        assert weights.shape[0] == self.filters                 , "ERROR (weights): invalid filter dimension"
        assert weights.shape[1] == self.channels//self.groups   , "ERROR (weights): invalid channel dimension"
        assert weights.shape[2] == self.kernel_size[0]          , "ERROR (weights): invalid kernel dimension"
        assert weights.shape[3] == self.kernel_size[1]          , "ERROR (weights): invalid kernel dimension"

        assert bias.shape[0] == self.filters, "ERROR (bias): invalid filter dimension"

        # instantiate convolution layer
        convolution_layer = torch.nn.Conv2d(self.channels_in(), self.filters,
                self.kernel_size, stride=self.stride, padding=0, groups=self.groups)

        # update weights
        convolution_layer.weight = torch.nn.Parameter(torch.from_numpy(weights))

        # update bias
        convolution_layer.bias = torch.nn.Parameter(torch.from_numpy(bias))

        # get the padding
        padding = [
            self.pad_left,
            self.pad_right,
            self.pad_top,
            self.pad_bottom
        ]

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        data = torch.nn.functional.pad(torch.from_numpy(data), padding, "constant", 0.0)
        return convolution_layer(data).detach().numpy()

@dataclass(kw_only=True)
class ConvolutionLayerTrait3D(ConvolutionLayerTrait2D):
    kernel_depth: int = 1
    stride_depth: int = 1
    pad_front: int = 0
    pad_back: int = 0

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        # TODO: change the module classes to their 3D counterparts


    @property
    def kernel_size(self) -> List[int]:
        return [ self.kernel_rows, self.kernel_cols, self.kernel_depth ]

    @property
    def stride(self) -> List[int]:
        return [ self.stride_rows, self.stride_cols, self.stride_depth ]

    @property
    def pad(self) -> List[int]:
        return [
            self.pad_top,
            self.pad_left,
            self.pad_front,
            self.pad_bottom,
            self.pad_right,
            self.pad_back,
        ]

    @kernel_size.setter
    def kernel_size(self, val: List[int]) -> None:
        assert(len(val) == 3, "kernel size must be a list of three integers")
        self.kernel_rows    = val[0]
        self.kernel_cols    = val[1]
        self.kernel_depth   = val[2]

    @stride.setter
    def stride(self, val: List[int]) -> None:
        assert(len(val) == 3, "stride must be a list of three integers")
        self.stride_rows    = val[0]
        self.stride_cols    = val[1]
        self.stride_depth   = val[2]

    @pad.setter
    def pad(self, val: List[int]) -> None:
        assert(len(val) == 6, "pad must be a list of six integers")
        self.pad_top    = val[0]
        self.pad_right  = val[4]
        self.pad_bottom = val[3]
        self.pad_left   = val[1]
        self.pad_front  = val[2]
        self.pad_back   = val[5]

    def depth_out(self) -> int:
        return self.modules["sliding_window"].depth_out()

    def update_modules(self):

        # update all the existing modules
        super().update_modules()

        # iterate over the modules and update depth
        for module in self.modules:
            match module:
                case "sliding_window":
                    self.modules[module].depth = self.depth_in()
                case _:
                    self.modules[module].depth = self.depth_out()


