import importlib
import math
from typing import Union, List
from dataclasses import dataclass, field

import pydot
import numpy as np
from dacite import from_dict

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model
from fpgaconvnet.models.layers import Layer

from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY, SPARSITY


@dataclass(kw_only=True)
class ConvolutionLayerBase(Layer):

    @classmethod
    def build_from_arch(cls, arch: Architecture, conf: dict):

        # a list for all the base classes
        base_classes = [cls]

        # add the backend base class
        match arch.backend:
            case BACKEND.HLS:
                base_classes.append(ConvolutionLayerHLS)
            case BACKEND.CHISEL:
                base_classes.append(ConvolutionLayerChisel)
            case _:
                raise ValueError(f"Invalid backend {arch.backend}")

        # optionally add the dimensionality base
        if arch.dimensionality == DIMENSIONALITY.THREE:
                base_classes.append(ConvolutionLayer3D)

        # optionally add a sparsity base class
        if arch.sparsity == SPARSITY.SPARSE:

            # firstly, make sure it uses the chisel backend
            assert arch.backend == BACKEND.CHISEL, "Sparse layers are only supported in the chisel backend"

            # get the kernel size product
            kernel_size = np.prod([ conf.get(key, 1) for key in ["kernel_rows", "kernel_cols", "kernel_depth"] ])

            if kernel_size == 1:
                base_classes.append(ConvolutionLayerSparsePointwise)
            else:
                base_classes.append(ConvolutionLayerSparse)

        # create a new type that inherits from all the base classes
        ## this is inspired by https://stackoverflow.com/a/21061856
        convolution_type = dataclass(kw_only=True)(type("ConvolutionLayer"+str(arch), tuple(reversed(base_classes)), {}))

        # create an instance of the new convolution type
        return from_dict(data_class=convolution_type, data=conf)


    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        print("Convolution Layer Base")

@dataclass(kw_only=True)
class ConvolutionLayerHLS:
    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        print("Convolution Layer HLS")

@dataclass(kw_only=True)
class ConvolutionLayerChisel:
    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        print("Convolution Layer Chisel")

@dataclass(kw_only=True)
class ConvolutionLayer3D:

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        print("Convolution Layer 3D")

@dataclass(kw_only=True)
class ConvolutionLayerSparse:

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        print("Convolution Layer Sparse")

@dataclass(kw_only=True)
class ConvolutionLayerSparsePointwise:

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        print("Convolution Layer Sparse Pointwise")

