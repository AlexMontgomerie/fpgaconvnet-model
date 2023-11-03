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
from .dimensions import ConvolutionLayerTrait2D, ConvolutionLayerTrait3D
from .sparse import ConvolutionLayerTraitSparse, ConvolutionLayerTraitSparsePointwise

from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

class ConvolutionLayerBaseMeta(LayerBaseMeta):

    @classmethod
    def build_type_from_arch(cls, arch: Architecture, conf: dict):

        # a list for all the base classes
        base_classes = [Layer, ConvolutionLayerBase]

        # add the backend base class
        match arch.backend:
            case BACKEND.HLS:
                base_classes.append(ConvolutionLayerTraitHLS)
            case BACKEND.CHISEL:
                base_classes.append(ConvolutionLayerTraitChisel)
            case _:
                raise ValueError(f"Invalid backend {arch.backend}")

        # add the dimensionality base
        if arch.dimensionality == DIMENSIONALITY.THREE:
            base_classes.extend([Layer3D, ConvolutionLayerTrait3D])
        else:
            base_classes.extend([ConvolutionLayerTrait2D])

        # optionally add a sparsity base class
        if arch.sparsity == SPARSITY.SPARSE:

            # firstly, make sure it uses the chisel backend
            assert arch.backend == BACKEND.CHISEL, "Sparse layers are only supported in the chisel backend"

            # get the kernel size product
            kernel_size = np.prod([ conf.get(key, 1) for key in ["kernel_rows", "kernel_cols", "kernel_depth"] ])

            if kernel_size == 1:
                # choose a pointwise version
                base_classes.append(ConvolutionLayerTraitSparsePointwise)
            else:
                # choose a generic sparse version
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

