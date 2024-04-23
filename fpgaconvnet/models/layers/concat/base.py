import math
from typing import ClassVar, Any
from dataclasses import dataclass
from collections import OrderedDict

import numpy as np
from dacite import from_dict
import networkx as nx # type: ignore

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
class ConcatLayerBase(LayerMatchingCoarse, LayerBase):
    ports: int
    channels: list[int]
    data_t: FixedPoint = FixedPoint(16,8)

    name: ClassVar[str] = "concat"

    def __post_init__(self):


        assert self.ports == len(self.channels), "ERROR: number of ports must match number of channels"
        assert self.ports > 1, "ERROR: number of ports must be greater than 1"

        # update ports_in and ports_out
        self.ports_in = self.ports
        self.ports_out = 1

        # call the parent post_init
        super().__post_init__()

    def __setattr__(self, name: str, value: Any) -> None:

        if not hasattr(self, "is_init"):
            super().__setattr__(name, value)
            return

        try:
            match name:
                case "ports" | "ports_in":
                    super().__setattr__("ports", value)
                    super().__setattr__("ports_in", value)
                    super().__setattr__("ports_out", 1)

                case "ports_out":
                    raise ValueError("ERROR: cannot set ports_out (always 1)")

                case _:
                    super().__setattr__(name, value)

        except AttributeError:
            print(f"WARNING: unable to set attribute {name}, trying super method")
            super().__setattr__(name, value)


    def get_coarse_feasible(self) -> list[int]:
        coarse_in_feasible = super().get_coarse_in_feasible()
        coarse_out_feasible = super().get_coarse_out_feasible()
        return list(set(coarse_in_feasible).intersection(set(coarse_out_feasible)))

    def get_coarse_in_feasible(self) -> list[int]:
        return self.get_coarse_feasible()

    def get_coarse_out_feasible(self) -> list[int]:
        return self.get_coarse_feasible()

    def functional_model(self, *data: np.array) -> np.array:
        import torch

        assert len(data) == self.ports, f"invalid number of input ports ({len(data)} != {self.ports})"

        for i, d in enumerate(data):
            assert list(d.shape) == self.input_shape(i), \
                    f"invalid spatial dimensions for port={i} ({list(d.shape)} != {self.input_shape(i)})"

        # return the functional model
        return torch.cat([torch.from_numpy(d) for d in data], axis=-1)

    def layer_info(self, parameters, batch_size=1):
        super().layer_info(parameters, batch_size)
        parameters.ports = self.ports
        parameters.channels_in_array.extend(self.channels)

@dataclass(kw_only=True)
class ConcatLayerChiselMixin(ConcatLayerBase):

    backend: ClassVar[BACKEND] = BACKEND.CHISEL

    @property
    def module_lookup(self) -> dict:
        return OrderedDict({
            "concat": self.get_concat_parameters,
        })

    def get_concat_parameters(self) -> dict:
        return {
            "repetitions": math.prod(self.input_shape()[:-1]),
            "ports": self.ports,
            "channels": [ c//self.streams() for c in self.channels ],
            "streams": self.coarse,
            "data_t": self.data_t,
        }

    def build_module_graph(self) -> nx.DiGraph:

        # get the module graph
        self.graph = nx.DiGraph()

        # add the concat module
        self.graph.add_node("concat", module=self.modules["concat"])


@dataclass(kw_only=True)
class ConcatLayer2DMixin(ConcatLayerBase, MultiPortLayer2D):
    rows: int # type: ignore
    cols: int # type: ignore
    channels: list[int]

    def rows_in(self, port_idx: int = 0) -> int:
        assert port_idx < self.ports_in
        return self.rows

    def cols_in(self, port_idx: int = 0) -> int:
        assert port_idx < self.ports_in
        return self.cols

    def channels_in(self, port_idx: int = 0) -> int:
        assert port_idx < self.ports_in
        return self.channels[port_idx]

    def rows_out(self, port_idx: int = 0) -> int:
        assert port_idx == 0
        return self.rows

    def cols_out(self, port_idx: int = 0) -> int:
        assert port_idx == 0
        return self.cols

    def channels_out(self, port_idx: int = 0) -> int:
        assert port_idx == 0
        return sum(self.channels)
