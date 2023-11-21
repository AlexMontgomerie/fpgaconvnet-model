import importlib
import math
from typing import Union, ClassVar
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from collections import OrderedDict

import pydot
import numpy as np
from dacite import from_dict

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

from fpgaconvnet.models.layers import LayerBase, Layer2D, Layer3D
from fpgaconvnet.models.layers.traits import LayerMatchingCoarse
from fpgaconvnet.models.modules import ModuleBase

from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

@dataclass(kw_only=True)
class SplitLayerBase(LayerMatchingCoarse, LayerBase):

    ports: int
    data_t: FixedPoint = FixedPoint(16,8)

    name: ClassVar[str] = "relu"

    @property
    def module_lookup(self) -> dict:
        return OrderedDict({
            "fork": self.get_fork_parameters,
        })

    def get_fork_parameters(self) -> dict:

        match self.backend:

            case BACKEND.HLS:
                return {
                    **self.input_shape_dict(),
                    "channels": self.channels//self.coarse,
                    "data_t": self.data_t,
                }

            case BACKEND.CHISEL:
                return {
                    "repetitions": int(np.prod(self.input_shape()))//self.coarse,
                    "streams": self.coarse,
                    "fine": 1,
                    "coarse": self.ports,
                    "data_t": self.data_t,
                }

class SplitLayerChisel(Layer2D, SplitLayerBase):
    backend: ClassVar[BACKEND] = BACKEND.CHISEL
    register: ClassVar[bool] = True

class SplitLayerHLS(SplitLayerBase, Layer2D):
    backend: ClassVar[BACKEND] = BACKEND.HLS
    register: ClassVar[bool] = True

class SplitLayerHLS3D(SplitLayerBase, Layer3D):
    backend: ClassVar[BACKEND] = BACKEND.HLS
    register: ClassVar[bool] = True

