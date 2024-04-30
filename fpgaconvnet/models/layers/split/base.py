import math
from typing import ClassVar, Any
from dataclasses import dataclass, field
from collections import OrderedDict
from overrides import override

import numpy as np
from dacite import from_dict
import networkx as nx

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

from fpgaconvnet.models.layers import LayerBase
from fpgaconvnet.models.layers.traits import LayerMatchingCoarse, MultiPortLayer2D
from fpgaconvnet.models.modules import ModuleBase

from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

@dataclass(kw_only=True)
class SplitLayerBase(LayerMatchingCoarse, LayerBase):

    ports: int
    data_t: FixedPoint = field(default_factory=lambda: FixedPoint(16, 8))

    name: ClassVar[str] = "split"

    def __post_init__(self):

        assert self.ports > 1, "ERROR: number of ports must be greater than 1"

        # update ports_in and ports_out
        self.ports_in =  1
        self.ports_out = self.ports

        # call the parent post_init
        super().__post_init__()

    def __setattr__(self, name: str, value: Any) -> None:

        if not hasattr(self, "is_init"):
            super().__setattr__(name, value)
            return

        try:
            match name:
                case "ports" | "ports_out":
                    super().__setattr__("ports", value)
                    super().__setattr__("ports_out", value)
                    super().__setattr__("ports_in", 1)

                case "ports_in":
                    raise ValueError("ERROR: cannot set ports_out (always 1)")

                case _:
                    super().__setattr__(name, value)

        except AttributeError:
            print(f"WARNING: unable to set attribute {name}, trying super method")
            super().__setattr__(name, value)

    def functional_model(self,data,batch_size=1):
        import torch

        assert list(data.shape) == self.input_shape(), \
                f"invalid input shape dimension ({data.shape} != {self.input_shape()})"

        # duplicate the input data
        return [data] * self.ports

    def layer_info(self, parameters, batch_size=1):
        super().layer_info(parameters, batch_size)
        parameters.ports = self.ports

@dataclass(kw_only=True)
class SplitLayerChiselMixin(SplitLayerBase):

    backend: ClassVar[BACKEND] = BACKEND.CHISEL

    @property
    def module_lookup(self) -> dict:
        return OrderedDict({
            "fork": self.get_fork_parameters,
        })

    def get_fork_parameters(self) -> dict:
        return {
            "repetitions": math.prod(self.input_shape())//self.streams(),
            "coarse": self.ports,
            "streams": self.coarse,
            "data_t": self.data_t,
            "is_sync": True,
            "fine": 1,
        }

    def build_module_graph(self) -> nx.DiGraph:

        # get the module graph
        self.module_graph = nx.DiGraph()

        # add the split module
        self.module_graph.add_node("fork", module=self.modules["fork"])


@dataclass(kw_only=True)
class SplitLayer2DMixin(SplitLayerBase, MultiPortLayer2D):
    rows: int
    cols: int
    channels: int

    @override
    def rows_in(self, port_idx: int = 0) -> int:
        assert port_idx == 0
        return self.rows

    @override
    def cols_in(self, port_idx: int = 0) -> int:
        assert port_idx == 0
        return self.cols

    @override
    def channels_in(self, port_idx: int = 0) -> int:
        assert port_idx == 0
        return self.channels

    @override
    def rows_out(self, port_idx: int = 0) -> int:
        assert port_idx < self.ports_out
        return self.rows

    @override
    def cols_out(self, port_idx: int = 0) -> int:
        assert port_idx < self.ports_out
        return self.cols

    @override
    def channels_out(self, port_idx: int = 0) -> int:
        assert port_idx < self.ports_out
        return self.channels

