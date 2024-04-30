from typing import ClassVar, Optional
from abc import abstractmethod
from dataclasses import dataclass, field
from overrides import override
import math

import pydot
import numpy as np
from dacite import from_dict

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model
from fpgaconvnet.models.modules.resources import ResourceModel

from fpgaconvnet.models.layers import LayerBase
from fpgaconvnet.models.layers.traits import Layer2D, Layer3D
from fpgaconvnet.models.modules import *

from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

@dataclass(kw_only=True)
class InnerProductLayerBase(LayerBase):
    filters: int
    input_t: FixedPoint = field(default_factory=lambda: FixedPoint(16,8))
    output_t: FixedPoint = field(default_factory=lambda: FixedPoint(16,8))
    weight_t: FixedPoint = field(default_factory=lambda: FixedPoint(16,8))
    acc_t: FixedPoint = field(default_factory=lambda: FixedPoint(32,16))
    weight_compression_ratio: list = field(default_factory=lambda: [1.0], init=True)

    name: ClassVar[str] = "inner_product"

    def __post_init__(self):

        # call the super init
        super().__post_init__()

        # off chip weight streaming attributes
        self.weight_array_unit_depth = 0
        self.weight_array_unit_width = 0

    def get_weights_reloading_feasible(self) -> list[int]:
        return get_factors(self.filters // self.coarse_out)

    def get_parameters_size(self) -> dict:
        weights_size = self.channels_in() * self.filters
        bias_size = 0
        return {
            "weights"   : weights_size,
            "bias"      : bias_size
        }

    def get_operations(self) -> int:
        ops = self.channels_in()*math.prod(self.shape_out)
        if self.has_bias:
            ops += math.prod(self.shape_out)
        return ops

    def get_weight_memory_depth(self) -> int:
        return self.filters*self.channels_in() // (self.coarse_in*self.coarse_out)

    def get_weight_resources(self) -> (int, int):

    # get the depth for the weights memory
        weight_memory_depth = self.get_weight_memory_depth()

        bram_rsc = bram_array_resource_model(weight_memory_depth, self.weight_t.width, "memory") * self.coarse_in*self.coarse_out

        # return the memory resource model
        return bram_rsc, 0  # (bram usage, uram usage)

    @override
    def resource(self, model: Optional[ResourceModel] = None) -> dict[str,int]:

        # get the module resources
        rsc = super().resource(model)

        # get the weights resources
        weights_bram, weights_uram = self.get_weight_resources()
        rsc["BRAM"] += weights_bram
        rsc["URAM"] = weights_uram

        # return the resource usage
        return rsc

    def functional_model(self,data,weights,bias,batch_size=1):
        import torch

        assert list(data.shape) == self.input_shape(0), \
                f"invalid spatial dimensions ({list(data.shape)} != {self.input_shape(0)})"

        assert weights.shape[0] == self.filters, "ERROR (weights): invalid filter dimension"
        assert weights.shape[1] == math.prod(self.input_shape(0)), \
                "ERROR (weights): invalid channel dimension"

        assert bias.shape[0] == self.filters, "ERROR (bias): invalid filter dimension"

        # instantiate inner product layer
        inner_product_layer = torch.nn.Linear(
                math.prod(self.input_shape(0)), self.filters)#, bias=False)

        # update weights
        inner_product_layer.weight = torch.nn.Parameter(torch.from_numpy(weights))

        # update bias
        inner_product_layer.bias = torch.nn.Parameter(torch.from_numpy(bias))

        # return output featuremap
        data = np.moveaxis(data, -1, 0).flatten()
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return inner_product_layer(torch.from_numpy(data)).detach().numpy()

    def layer_info(self, parameters, batch_size=1):
        super().layer_info(parameters, batch_size)
        parameters.filters = self.filters
        self.input_t.to_protobuf(parameters.input_t)
        self.weight_t.to_protobuf(parameters.weight_t)
        self.acc_t.to_protobuf(parameters.acc_t)
        self.output_t.to_protobuf(parameters.output_t)


@dataclass(kw_only=True)
class InnerProductLayer2DMixin(InnerProductLayerBase, Layer2D):

    def rows_in(self) -> int:
        return self.rows

    def cols_in(self) -> int:
        return self.cols

    def channels_in(self) -> int:
        return self.channels

    def rows_out(self) -> int:
        return 1

    def cols_out(self) -> int:
        return 1

    def channels_out(self) -> int:
        return self.filters

@dataclass(kw_only=True)
class InnerProductLayer3DMixin(Layer3D, InnerProductLayer2DMixin):

    def depth_in(self) -> int:
        return self.depth

    def depth_out(self) -> int:
        return 1

