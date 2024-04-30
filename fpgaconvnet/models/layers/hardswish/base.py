import math
from typing import ClassVar
from dataclasses import dataclass, field
from collections import OrderedDict
from abc import abstractmethod

import pydot # type: ignore
import numpy as np
from dacite import from_dict
import networkx as nx # type: ignore

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

from fpgaconvnet.models.layers import LayerBase
from fpgaconvnet.models.layers.traits import LayerMatchingCoarse
from fpgaconvnet.models.modules import ModuleBase

from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

@dataclass(kw_only=True)
class HardswishLayerBase(LayerMatchingCoarse, LayerBase):

    input_t: FixedPoint = field(default_factory=lambda: FixedPoint(16,8))
    output_t: FixedPoint = field(default_factory=lambda: FixedPoint(16,8))

    name: ClassVar[str] = "hardswish"

    def functional_model(self,data,batch_size=1):
        import torch

        assert data.shape == self.inputs_shape(), "ERROR: invalid input shape dimension"

        # instantiate hardswish layer
        hardswish_layer = torch.nn.Hardswish()

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return hardswish_layer(torch.from_numpy(data)).detach().numpy()

class HardswishLayerChiselMixin(HardswishLayerBase):

    backend: ClassVar[BACKEND] = BACKEND.CHISEL

    @property
    def module_lookup(self) -> dict:
        return OrderedDict({
            "hardswish": self.get_hardswish_parameters,
        })

    def get_hardswish_parameters(self) -> dict:
        return {
            "repetitions": math.prod(self.input_shape())//self.streams(),
            "streams": self.coarse,
            "input_t": self.input_t,
            "output_t": self.output_t,
        }

    def build_module_graph(self) -> nx.DiGraph:

        # get the module graph
        self.module_graph = nx.DiGraph()

        # add the hardswish module
        self.module_graph.add_node("hardswish", module=self.modules["hardswish"])


class HardswishLayerHLSMixin(HardswishLayerBase):

    backend: ClassVar[BACKEND] = BACKEND.HLS

    @property
    def module_lookup(self) -> dict:
        return OrderedDict({
            "hardswish": self.get_hardswish_parameters,
        })

    def get_hardswish_parameters(self) -> dict:
        return {
            **self.input_shape_dict(),
            "channels": self.channels//self.coarse, # type: ignore
            "input_t": self.input_t,
            "output_t": self.output_t,
        }

    def build_module_graph(self) -> nx.DiGraph:

        # get the module graph
        self.module_graph = nx.DiGraph()

        # add the hardswish module
        for i in range(self.coarse):
            self.module_graph.add_node(f"hardswish_{i}", module=self.modules["hardswish"])

