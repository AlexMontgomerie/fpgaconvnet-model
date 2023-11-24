import math
from typing import ClassVar
from dataclasses import dataclass
from collections import OrderedDict
from overrides import override
import pydot
import numpy as np
from dacite import from_dict
import networkx as nx

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

from fpgaconvnet.models.layers import LayerBase
from fpgaconvnet.models.layers.traits import LayerMatchingCoarse, Layer2D, Layer3D
from fpgaconvnet.models.modules import ModuleBase

from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

@dataclass(kw_only=True)
class GlobalPoolingLayerBase(LayerMatchingCoarse, LayerBase):

    op_type: str = "avg"
    acc_t: FixedPoint = FixedPoint(32,16)
    data_t: FixedPoint = FixedPoint(16,8)

    name: ClassVar[str] = "global_pool"

    def get_operations(self):
        return math.prod(self.input_shape())

    def functional_model(self, data):

        assert data.shape == self.inputs_shape(), "ERROR: invalid input shape dimension"

        # return output featuremap
        return np.average(data, axis=list(range(len(data.shape)-1)))

class GlobalPoolingLayerChiselMixin(GlobalPoolingLayerBase):

    backend: ClassVar[BACKEND] = BACKEND.CHISEL

    @property
    def module_lookup(self) -> dict:
        return OrderedDict({
            "global_pool": self.get_global_pool_parameters,
        })

    def get_global_pool_parameters(self) -> dict:
        return {
            "repetitions": 1,
            **self.input_shape_dict(),
            "channels": self.channels_in()//self.streams(),
            "streams": self.coarse,
            "data_t": self.data_t,
            "acc_t": self.acc_t,
        }

    def build_module_graph(self) -> nx.DiGraph:

        # get the module graph
        self.graph = nx.DiGraph()

        # add the global_pool module
        self.graph.add_node("global_pool", module=self.modules["global_pool"])

class GlobalPoolingLayer2DMixin(GlobalPoolingLayerBase, Layer2D):

    @override
    def rows_out(self) -> int:
        return 1

    @override
    def cols_out(self) -> int:
        return 1

class GlobalPoolingLayer3DMixin(GlobalPoolingLayerBase, Layer3D):

    @override
    def depth_out(self) -> int:
        return 1


