import math
from typing import ClassVar
from dataclasses import dataclass
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
from fpgaconvnet.models.layers.traits import LayerMatchingCoarse, Layer2D, Layer3D
from fpgaconvnet.models.modules import ModuleBase

from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

@dataclass(kw_only=True)
class ResizeLayerBase(LayerMatchingCoarse, LayerBase):

    scales: list[int]
    mode: str = "nearest"
    data_t: FixedPoint = FixedPoint(16,8)

    name: ClassVar[str] = "resize"

    def get_operations(self):
        return math.prod(self.input_shape())


    def functional_model(self, data, batch_size=1):
        import torch

        assert list(data.shape) == self.input_shape(), \
            f"invalid input shape dimension ({data.shape} != {self.input_shape()})"

        # instantiate resize layer
        resize_layer = torch.nn.functional.interpolate

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return resize_layer(torch.from_numpy(data),
                            scale_factor=tuple(self.scales[:-1]),
                            mode=self.mode).detach().numpy()

    def layer_info(self, parameters, batch_size=1):
        super().layer_info(parameters, batch_size)
        parameters.scales.extend(self.scales)

class ResizeLayerChiselMixin(ResizeLayerBase):

    backend: ClassVar[BACKEND] = BACKEND.CHISEL

    @property
    def module_lookup(self) -> dict:
        return OrderedDict({
            "resize": self.get_resize_parameters,
        })

    def get_resize_parameters(self) -> dict:
        return {
            "repetitions": 1,
            **self.input_shape_dict(),
            "channels": self.channels_in()//self.streams(),
            "scales": self.scales,
            "streams": self.coarse,
            "data_t": self.data_t,
        }

    def build_module_graph(self) -> nx.DiGraph:

        # get the module graph
        self.graph = nx.DiGraph()

        # add the resize module
        self.graph.add_node("resize", module=self.modules["resize"])

class ResizeLayer2DMixin(ResizeLayerBase, Layer2D):

    def rows_out(self) -> int:
        return self.modules['resize'].rows_out

    def cols_out(self) -> int:
        return self.modules['resize'].cols_out

