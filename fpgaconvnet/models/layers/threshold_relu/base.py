import math
from typing import ClassVar
from dataclasses import dataclass
from collections import OrderedDict

import numpy as np
from dacite import from_dict
import networkx as nx

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
class ThresholdReLULayerBase(LayerMatchingCoarse, LayerBase):

    threshold: float = 0.0,
    data_t: FixedPoint = FixedPoint(16,8)

    name: ClassVar[str] = "threshold_relu"

    def functional_model(self,data,batch_size=1):
        import torch

        assert data.shape == self.inputs_shape(), "ERROR: invalid input shape dimension"

        # instantiate threshold_relu layer
        relu_layer = torch.nn.Threshold(self.threshold, 0.0)

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return relu_layer(torch.from_numpy(data)).detach().numpy()

class ThresholdReLULayerChiselMixin(ThresholdReLULayerBase):

    backend: ClassVar[BACKEND] = BACKEND.CHISEL

    @property
    def module_lookup(self) -> dict:
        return OrderedDict({
            "threshold_relu": self.get_threshold_relu_parameters,
        })

    def get_threshold_relu_parameters(self) -> dict:
        return {
            "repetitions": math.prod(self.input_shape())//self.streams(),
            "threshold": self.threshold,
            "streams": self.coarse,
            "data_t": self.data_t,
        }

    def build_module_graph(self) -> nx.DiGraph:

        # get the module graph
        self.graph = nx.DiGraph()

        # add the threshold_relu module
        self.graph.add_node("threshold_relu", module=self.modules["threshold_relu"])


class ThresholdReLULayerHLSMixin(ThresholdReLULayerBase):

    backend: ClassVar[BACKEND] = BACKEND.HLS

    @property
    def module_lookup(self) -> dict:
        return OrderedDict({
            "threshold_relu": self.get_threshold_relu_parameters,
        })

    def get_threshold_relu_parameters(self) -> dict:
        return {
            **self.input_shape_dict(),
            "threshold": self.threshold,
            "channels": self.channels//self.coarse,
            "data_t": self.data_t,
        }

    def build_module_graph(self) -> nx.DiGraph:

        # get the module graph
        self.graph = nx.DiGraph()

        # add the threshold_relu module
        for i in range(self.coarse):
            self.graph.add_node(f"threshold_relu_{i}", module=self.modules["threshold_relu"])

