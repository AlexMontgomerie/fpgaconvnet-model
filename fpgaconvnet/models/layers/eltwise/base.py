import math
from typing import ClassVar, Any
from dataclasses import dataclass
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
class EltwiseLayerBase(LayerMatchingCoarse, LayerBase):

    ports: int
    op_type: str = "add"
    broadcast: bool = False
    data_t: FixedPoint = FixedPoint(16,8)
    acc_t: FixedPoint = FixedPoint(32,16)

    name: ClassVar[str] = "eltwise"

    def __post_init__(self):
        self.ports_in = self.ports
        self.ports_out = 1
        # self.buffer_depth = [0]*self.ports
        self.buffer_depth = [0]*self.ports * 100
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

    def get_buffer_depth(self, port_idx: int = 0) -> int:
        assert port_idx < self.ports, \
                f"port_idx {port_idx} >= self.ports_in {self.ports}"
        return self.buffer_depth[port_idx]

    def layer_info(self, parameters, batch_size=1):
        super().layer_info(parameters, batch_size)
        parameters.ports = self.ports

    def functional_model(self, *data: np.array) -> np.array:
        import torch

        assert len(data) == self.ports, f"invalid number of input ports ({len(data)} != {self.ports})"

        for i, d in enumerate(data):
            assert list(d.shape) == self.input_shape(i), \
                    f"invalid spatial dimensions ({list(d.shape)} != {self.input_shape(i)})"

        # return the functional model
        return torch.add(torch.from_numpy(data[0]), torch.from_numpy(data[0])).detach().numpy()


    def layer_info(self, parameters, batch_size=1):
        super().layer_info(parameters, batch_size)
        parameters.ports = self.ports
        # parameters.op_type = self.op_type
        # parameters.broadcast = self.broadcast
        # parameters.data_t = self.data_t
        self.acc_t.to_protobuf(parameters.acc_t)


@dataclass(kw_only=True)
class EltwiseLayerChiselMixin(EltwiseLayerBase):

    backend: ClassVar[BACKEND] = BACKEND.CHISEL

    @property
    def module_lookup(self) -> dict:
        return OrderedDict({
            "eltwise": self.get_eltwise_parameters,
        })

    def get_eltwise_parameters(self) -> dict:
        return {
            "repetitions": math.prod(self.input_shape())//self.streams(),
            "eltwise_type": self.op_type,
            "broadcast": self.broadcast,
            "ports": self.ports,
            "streams": self.coarse,
            "data_t": self.data_t,
        }

    def build_module_graph(self) -> nx.DiGraph:

        # get the module graph
        self.graph = nx.DiGraph()

        # add the eltwise module
        self.graph.add_node("eltwise", module=self.modules["eltwise"])


@dataclass(kw_only=True)
class EltwiseLayer2DMixin(EltwiseLayerBase, MultiPortLayer2D):
    rows: int
    cols: int
    channels: int

    @override
    def rows_in(self, port_idx: int = 0) -> int:
        assert port_idx < self.ports_in
        return self.rows

    @override
    def cols_in(self, port_idx: int = 0) -> int:
        assert port_idx < self.ports_in
        return self.cols

    @override
    def channels_in(self, port_idx: int = 0) -> int:
        assert port_idx < self.ports_in
        return self.channels

    @override
    def rows_out(self, port_idx: int = 0) -> int:
        assert port_idx == 0
        return self.rows

    @override
    def cols_out(self, port_idx: int = 0) -> int:
        assert port_idx == 0
        return self.cols

    @override
    def channels_out(self, port_idx: int = 0) -> int:
        assert port_idx == 0
        return self.channels

