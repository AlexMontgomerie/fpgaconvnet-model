import importlib
import math
from typing import Any, List, Union
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field

import pydot
import numpy as np
from dacite import from_dict

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.data_types import FixedPoint

from fpgaconvnet.models.layers import LayerBaseMeta, Layer, Layer3D
from fpgaconvnet.models.modules import GlobalPooling

from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY, SPARSITY

from .backend import GlobalPoolingLayerTraitHLS, GlobalPoolingLayerTraitChisel
from .dimensions import GlobalPoolingLayerTrait2D, GlobalPoolingLayerTrait3D

class GlobalPoolingLayerBaseMeta(LayerBaseMeta):

    @classmethod
    def build_type_from_arch(cls, arch: Architecture, conf: dict):

        # a list for all the base classes
        base_classes = [Layer, GlobalPoolingLayerBase]

        # add the backend base class
        match arch.backend:
            case BACKEND.HLS:
                base_classes.append(GlobalPoolingLayerTraitHLS)
            case BACKEND.CHISEL:
                base_classes.append(GlobalPoolingLayerTraitChisel)
            case _:
                raise ValueError(f"Invalid backend {arch.backend}")

        # add the dimensionality base
        if arch.dimensionality == DIMENSIONALITY.THREE:
            base_classes.extend([Layer3D, GlobalPoolingLayerTrait3D])
        else:
            base_classes.extend([GlobalPoolingLayerTrait2D])

        # create a new type that inherits from all the base classes
        ## this is inspired by https://stackoverflow.com/a/21061856
        return dataclass(kw_only=True)(type("GlobalPoolingLayer"+str(arch), tuple(reversed(base_classes)), {}))

    @classmethod
    def build_from_arch(cls, arch: Architecture, conf: dict):

        # create the type from architecture and configuration
        convolution_type = cls.build_type_from_arch(arch, conf)

        # create an instance of the new convolution type
        return from_dict(data_class=convolution_type, data=conf)


@dataclass(kw_only=True)
class GlobalPoolingLayerBase(metaclass=GlobalPoolingLayerBaseMeta):
    coarse: int = 1
    acc_t: FixedPoint = field(default_factory=lambda: FixedPoint(32,16), init=True)
    op_type: str = "avg"
    regression_model: str = "linear_regression"

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        # get the backend
        match self.arch.backend:
            case BACKEND.HLS:
                backend = "hls"
            case BACKEND.CHISEL:
                backend = "chisel"
            case _:
                raise ValueError(f"Invalid backend {self.arch.backend}")

        # create all the modules
        self.modules["global_pooling"] = GlobalPooling(self.rows_in(),
                self.cols_in(), self.channels_in()//self.coarse, backend=backend,
                regression_model=self.regression_model)

        # update modules
        self.update_modules()

    def update_modules(self):

        # update the global_pooling module
        param = self.get_global_pooling_parameters()
        for p, v in param.items():
            setattr(self.modules["global_pooling"], p, v)

    def __setattr__(self, name: str, value: Any) -> None:

        if not hasattr(self, "is_init"):
            super().__setattr__(name, value)
            return

        match name:
            case "coarse" | "coarse_in" | "coarse_out":
                assert(value in self.get_coarse_in_feasible())
                assert(value in self.get_coarse_out_feasible())
                super().__setattr__("coarse_in", value)
                super().__setattr__("coarse_out", value)
                super().__setattr__("coarse", value)
                self.update()

            case _:
                super().__setattr__(name, value)

    def get_operations(self):
        return np.prod(self.shape_in)

    def rows_out(self) -> int:
        return 1

    def cols_out(self) -> int:
        return 1

    def resource(self):

        # get global_pooling3d resources
        global_pooling_rsc = self.modules['global_pooling'].rsc()

        # Total
        return {
            "LUT"  :  global_pooling_rsc['LUT']*self.coarse,
            "FF"   :  global_pooling_rsc['FF']*self.coarse,
            "BRAM" :  global_pooling_rsc['BRAM']*self.coarse,
            "DSP" :   global_pooling_rsc['DSP']*self.coarse,
        }


