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
from fpgaconvnet.models.modules import *

from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY, SPARSITY

from .backend import HardswishLayerTraitHLS, HardswishLayerTraitChisel
from .dimensions import HardswishLayerTrait2D, HardswishLayerTrait3D

class HardswishLayerBaseMeta(LayerBaseMeta):

    @classmethod
    def build_type_from_arch(cls, arch: Architecture, conf: dict):

        # a list for all the base classes
        base_classes = [Layer, HardswishLayerBase]

        # add the backend base class
        match arch.backend:
            case BACKEND.HLS:
                base_classes.append(HardswishLayerTraitHLS)
            case BACKEND.CHISEL:
                base_classes.append(HardswishLayerTraitChisel)
            case _:
                raise ValueError(f"Invalid backend {arch.backend}")

        # add the dimensionality base
        if arch.dimensionality == DIMENSIONALITY.THREE:
            base_classes.extend([Layer3D, HardswishLayerTrait3D])
        else:
            base_classes.extend([HardswishLayerTrait2D])

        # create a new type that inherits from all the base classes
        ## this is inspired by https://stackoverflow.com/a/21061856
        return dataclass(kw_only=True)(type("HardswishLayer"+str(arch), tuple(reversed(base_classes)), {}))

    @classmethod
    def build_from_arch(cls, arch: Architecture, conf: dict):

        # create the type from architecture and configuration
        convolution_type = cls.build_type_from_arch(arch, conf)

        # create an instance of the new convolution type
        return from_dict(data_class=convolution_type, data=conf)


@dataclass(kw_only=True)
class HardswishLayerBase(metaclass=HardswishLayerBaseMeta):
    coarse: int = 1
    input_t: FixedPoint = field(default_factory=lambda: FixedPoint(16,8), init=True)
    output_t: FixedPoint = field(default_factory=lambda: FixedPoint(16,8), init=True)
    regression_model: str = "linear_regression"

    @abstractmethod
    def update_modules(self):
        raise NotImplementedError

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

    @abstractmethod
    def resource(self) -> dict:
        raise NotImplementedError

