import math
from typing import ClassVar
from dataclasses import dataclass, field
from collections import OrderedDict

import pydot
import numpy as np
from dacite import from_dict
import networkx as nx

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

from fpgaconvnet.models.layers import LayerBase
from fpgaconvnet.models.modules import ModuleBase

from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

@dataclass(kw_only=True)
class SqueezeLayerBase(LayerBase):
    coarse_in: int
    coarse_out: int
    data_t: FixedPoint = field(default_factory=lambda: FixedPoint(16, 8))

    name: ClassVar[str] = "squeeze"

    def functional_model(self, data, batch_size=1):
        import torch

        assert list(data.shape) == self.input_shape(), \
                f"invalid input shape dimension ({data.shape} != {self.input_shape()})"

        # return the data as is
        return data

class SqueezeLayerChiselMixin(SqueezeLayerBase):

    backend: ClassVar[BACKEND] = BACKEND.CHISEL

    @property
    def module_lookup(self) -> dict:
        return OrderedDict({
            "squeeze": self.get_squeeze_parameters,
        })

    def get_squeeze_parameters(self) -> dict:
        return {
            "repetitions": math.prod(self.input_shape())//self.streams_in(),
            "coarse_in": self.coarse_in,
            "coarse_out": self.coarse_out,
            "streams": 1,
            "data_t": self.data_t,
        }

    def build_module_graph(self) -> nx.DiGraph:

        # get the module graph
        self.module_graph = nx.DiGraph()

        # add the squeeze module
        self.module_graph.add_node("squeeze", module=self.modules["squeeze"])

    # def rate_in(self, port_idx: int = 0) -> list[float]:
    #     assert port_idx == 0, "ERROR: invalid port index"
    #     return min(1.0, super().rate_in())

    # def rate_out(self, port_idx: int = 0) -> list[float]:
    #     assert port_idx == 0, "ERROR: invalid port index"
    #     return min(1.0, super().rate_out())

class SqueezeLayerHLSMixin(SqueezeLayerBase):

    backend: ClassVar[BACKEND] = BACKEND.HLS

    @property
    def module_lookup(self) -> dict:
        return OrderedDict({
            "squeeze": self.get_squeeze_parameters,
        })

    def get_squeeze_parameters(self) -> dict:
        return {
            **self.input_shape_dict(),
            "channels": self.channels//self.streams_in(),
            "coarse_in": self.coarse_in,
            "coarse_out": self.coarse_out,
            "data_t": self.data_t,
        }

    def build_module_graph(self) -> nx.DiGraph:

        # get the module graph
        self.module_graph = nx.DiGraph()

        # add the squeeze module
        self.module_graph.add_node(f"squeeze", module=self.modules["squeeze"])

