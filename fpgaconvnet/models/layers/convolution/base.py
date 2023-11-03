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

from .backend import ConvolutionLayerTraitHLS, ConvolutionLayerTraitChisel
from .sparse import ConvolutionLayerTraitSparse, ConvolutionLayerTraitSparsePointwise

from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

class ConvolutionLayerBaseMeta(LayerBaseMeta):

    @classmethod
    def build_type_from_arch(cls, arch: Architecture, conf: dict):

        # a list for all the base classes
        base_classes = []

        # add the dimensionality base
        if arch.dimensionality == DIMENSIONALITY.THREE:
            base_classes.extend([Layer3D, ConvolutionLayerBase, ConvolutionLayerTrait3D])
        else:
            base_classes.extend([Layer, ConvolutionLayerBase, ConvolutionLayerTrait2D])

        # add the backend base class
        match arch.backend:
            case BACKEND.HLS:
                base_classes.append(ConvolutionLayerTraitHLS)
            case BACKEND.CHISEL:
                base_classes.append(ConvolutionLayerTraitChisel)
            case _:
                raise ValueError(f"Invalid backend {arch.backend}")


        # optionally add a sparsity base class
        if arch.sparsity == SPARSITY.SPARSE:

            # firstly, make sure it uses the chisel backend
            assert arch.backend == BACKEND.CHISEL, "Sparse layers are only supported in the chisel backend"

            # get the kernel size product
            kernel_size = np.prod([ conf.get(key, 1) for key in ["kernel_rows", "kernel_cols", "kernel_depth"] ])

            if kernel_size == 1:
                base_classes.append(ConvolutionLayerTraitSparsePointwise)
            else:
                base_classes.append(ConvolutionLayerTraitSparse)

        # create a new type that inherits from all the base classes
        ## this is inspired by https://stackoverflow.com/a/21061856
        return dataclass(kw_only=True)(type("ConvolutionLayer"+str(arch), tuple(reversed(base_classes)), {}))

    @classmethod
    def build_from_arch(cls, arch: Architecture, conf: dict):

        # create the type from architecture and configuration
        convolution_type = cls.build_type_from_arch(arch, conf)

        # create an instance of the new convolution type
        return from_dict(data_class=convolution_type, data=conf)


@dataclass(kw_only=True)
class ConvolutionLayerBase(metaclass=ConvolutionLayerBaseMeta):
    filters: int
    groups: int = 1
    coarse_group: int = 1
    fine: int  = 1
    input_t: FixedPoint = FixedPoint(16,8)
    output_t: FixedPoint = FixedPoint(16,8)
    weight_t: FixedPoint = FixedPoint(16,8)
    acc_t: FixedPoint = FixedPoint(32,16)
    has_bias: int = 0
    stream_weights: int = 0
    use_uram: bool = False
    regression_model: str = "linear_regression"


    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        print("here")

        # check if the layer is depthwise
        self.depthwise = (self.groups == self.channels) and (self.groups == self.filters)
        self.pointwise = np.prod(self.kernel_size) == 1


    @abstractmethod
    def update_modules(self):
        raise NotImplementedError

    def __setattr__(self, name, value):

        if not hasattr(self, "is_init"):
            super().__setattr__(name, value)
            return

        match name:
            case "groups":
                assert(value in get_factors(self.channels_in()))
                super().__setattr__(name, value)

            case "coarse_group":
                assert(value in self.get_coarse_group_feasible())
                super().__setattr__(name, value)
                self.update()

            case "fine":
                assert(value in self.get_fine_feasible())
                super().__setattr__(name, value)
                self.update()

            case _:
                super().__setattr__(name, value)

    def channels_out(self) -> int:
        return self.filters

    def streams_in(self) -> int:
        return self.coarse_in*self.coarse_group

    def streams_out(self) -> int:
        return self.coarse_out*self.coarse_group

    def get_coarse_group_feasible(self) -> List[int]:
        return get_factors(self.groups)

    def get_coarse_in_feasible(self) -> List[int]:
        return get_factors(int(self.channels_in())//self.groups)

    def get_coarse_out_feasible(self) -> List[int]:
        return get_factors(int(self.channels_out())//self.groups)

    @abstractmethod
    def get_fine_feasible(self) -> List[int]:
        raise NotImplementedError

    def get_weights_reloading_feasible(self) -> List[int]:
        return get_factors(self.filters//(self.groups*self.coarse_out))

    def get_parameters_size(self) -> dict:
        weights_size = self.channels_in() * ( self.filters // self.groups ) * np.prod(self.kernel_size)
        bias_size = 0
        return {
            "weights"   : weights_size,
            "bias"      : bias_size
        }

    def get_operations(self) -> int:
        ops = np.prod(self.kernel_size)*self.channels_in()*np.prod(self.shape_out)
        if self.has_bias:
            ops += np.prod(self.shape_out)
        return ops

    def get_weight_memory_depth(self) -> int:
        return (self.filters//self.groups)*self.channels_in()*np.prod(self.kernel_size)// \
                                (self.fine*self.coarse_in*self.coarse_out*self.coarse_group)

    def get_weight_resources(self) -> (int, int):

    # get the depth for the weights memory
        weight_memory_depth = self.get_weight_memory_depth()

        bram_rsc = bram_array_resource_model(weight_memory_depth, self.weight_t.width, "memory") *\
            self.fine*self.coarse_in*self.coarse_out*self.coarse_group

        # return the memory resource model
        return bram_rsc, 0  # (bram usage, uram usage)

    @abstractmethod
    def resource(self) -> dict:
        raise NotImplementedError

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


