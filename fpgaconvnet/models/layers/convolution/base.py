from typing import ClassVar, Optional
from abc import abstractmethod
from dataclasses import dataclass
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
class ConvolutionLayerBase(LayerBase):
    filters: int
    groups: int = 1
    coarse_group: int = 1
    fine: int  = 1
    input_t: FixedPoint = FixedPoint(16,8)
    output_t: FixedPoint = FixedPoint(16,8)
    weight_t: FixedPoint = FixedPoint(16,8)
    acc_t: FixedPoint = FixedPoint(32,16)

    name: ClassVar[str] = "convolution"

    def __setattr__(self, name, value):

        if not hasattr(self, "is_init"):
            super().__setattr__(name, value)
            return

        match name:
            case "groups":
                assert(value in get_factors(self.channels))
                assert(value in get_factors(self.filters))
                super().__setattr__(name, value)

            case "coarse_group":
                assert(value in self.get_coarse_group_feasible())
                super().__setattr__(name, value)
                self.update()

            case "fine":
                assert(value in self.get_fine_feasible())
                super().__setattr__(name, value)
                self.update()

            case _:
                super().__setattr__(name, value)

    @property
    def pointwise(self) -> bool:
        return math.prod(self.kernel_size) == 1

    @property
    def depthwise(self) -> bool:
        return self.groups == self.channels and self.groups == self.filters

    @property
    @abstractmethod
    def kernel_size(self) -> list[int]:
        pass

    @property
    @abstractmethod
    def stride(self) -> list[int]:
        pass

    @property
    @abstractmethod
    def pad(self) -> list[int]:
        pass

    def streams_in(self) -> int:
        return int(self.coarse_in*self.coarse_group)

    def streams_out(self) -> int:
        return int(self.coarse_out*self.coarse_group)

    def get_coarse_group_feasible(self) -> list[int]:
        return get_factors(self.groups)

    def get_coarse_in_feasible(self) -> list[int]:
        return get_factors(self.channels_in() // self.groups)

    def get_coarse_out_feasible(self) -> list[int]:
        return get_factors(self.channels_out() // self.groups)

    @abstractmethod
    def get_fine_feasible(self) -> list[int]:
        pass

    def get_weights_reloading_feasible(self) -> list[int]:
        return get_factors(self.filters // (self.groups*self.coarse_out))

    def get_parameters_size(self) -> dict:
        weights_size = self.channels_in() * ( self.filters // self.groups ) * math.prod(self.kernel_size)
        bias_size = 0
        return {
            "weights"   : weights_size,
            "bias"      : bias_size
        }

    def get_operations(self) -> int:
        ops = math.prod(self.kernel_size)*self.channels_in()*math.prod(self.shape_out)
        if self.has_bias:
            ops += math.prod(self.shape_out)
        return ops

    def get_weight_memory_depth(self) -> int:
        return (self.filters//self.groups)*self.channels_in()*math.prod(self.kernel_size)// \
                                (self.fine*self.coarse_in*self.coarse_out*self.coarse_group)

    def get_weight_resources(self) -> (int, int):

    # get the depth for the weights memory
        weight_memory_depth = self.get_weight_memory_depth()

        bram_rsc = bram_array_resource_model(weight_memory_depth, self.weight_t.width, "memory") *\
            self.fine*self.coarse_in*self.coarse_out*self.coarse_group

        # return the memory resource model
        return bram_rsc, 0  # (bram usage, uram usage)

    @override
    def resource(self, model: Optional[ResourceModel] = None):

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

        assert data.shape == self.input_shape, "ERROR (data): invalid row dimension"

        assert weights.shape[0] == self.filters, "ERROR (weights): invalid filter dimension"
        assert weights.shape[1] == self.channels//self.groups, "ERROR (weights): invalid channel dimension"
        assert weights.shape[2:] == self.kernel_size, "ERROR (weights): invalid kernel dimension"

        assert bias.shape[0] == self.filters, "ERROR (bias): invalid filter dimension"

        # instantiate convolution layer
        convolution_layer = torch.nn.Conv2d(self.channels_in(), self.filters,
                self.kernel_size, stride=self.stride, padding=0, groups=self.groups)

        # update weights
        convolution_layer.weight = torch.nn.Parameter(torch.from_numpy(weights))

        # update bias
        convolution_layer.bias = torch.nn.Parameter(torch.from_numpy(bias))

        # # get the padding
        # padding = [
        #     self.pad_left,
        #     self.pad_right,
        #     self.pad_top,
        #     self.pad_bottom
        # ]

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        data = torch.nn.functional.pad(torch.from_numpy(data), self.pad, "constant", 0.0)
        return convolution_layer(data).detach().numpy()


@dataclass(kw_only=True)
class ConvolutionLayer2DBase(ConvolutionLayerBase, Layer2D):
    kernel_rows: int = 1
    kernel_cols: int = 1
    stride_rows: int = 1
    stride_cols: int = 1
    pad_top: int = 0
    pad_right: int = 0
    pad_bottom: int = 0
    pad_left: int = 0

    @property
    def kernel_size(self) -> list[int]:
        return [ self.kernel_rows, self.kernel_cols ]

    @property
    def stride(self) -> list[int]:
        return [ self.stride_rows, self.stride_cols ]

    @property
    def pad(self) -> list[int]: # TODO: change order
        return [
            self.pad_top,
            self.pad_left,
            self.pad_bottom,
            self.pad_right,
        ]

    @kernel_size.setter
    def kernel_size(self, val: list[int]) -> None:
        assert(len(val) == 2, "kernel size must be a list of two integers")
        self.kernel_rows = val[0]
        self.kernel_cols = val[1]

    @stride.setter
    def stride(self, val: list[int]) -> None:
        assert(len(val) == 2, "stride must be a list of two integers")
        self.stride_rows = val[0]
        self.stride_cols = val[1]

    @pad.setter
    def pad(self, val: list[int]) -> None:
        assert(len(val) == 4, "pad must be a list of four integers")
        self.pad_top    = val[0]
        self.pad_right  = val[3]
        self.pad_bottom = val[2]
        self.pad_left   = val[1]

    def rows_in(self) -> int:
        return self.rows

    def cols_in(self) -> int:
        return self.cols

    def channels_in(self) -> int:
        return self.channels

    def rows_out(self) -> int:
        assert "sliding_window" in self.modules.keys()
        return self.modules["sliding_window"].rows_out

    def cols_out(self) -> int:
        assert "sliding_window" in self.modules.keys()
        return self.modules["sliding_window"].cols_out

    def channels_out(self) -> int:
        return self.filters

    def pipeline_depth(self):
        # pipeline depth of the sliding window minus the total words in the pipeline from padding
        # plus the words needed to fill the accum buffer
        return (self.kernel_rows-1)*(self.cols+self.pad_left+self.pad_right)*self.channels//self.coarse_in + \
                (self.kernel_cols-1)*self.channels//self.coarse_in + \
                ((self.channels-1)//self.coarse_in)*(self.filters//(self.coarse_out*self.groups))

@dataclass(kw_only=True)
class ConvolutionLayer3DBase(Layer3D, ConvolutionLayer2DBase):
    kernel_depth: int = 1
    stride_depth: int = 1
    pad_front: int = 0
    pad_back: int = 0

    def depth_in(self) -> int:
        return self.depth

    def depth_out(self) -> int:
        assert "sliding_window" in self.modules.keys()
        return self.modules["sliding_window"].depth_out

    @property
    def kernel_size(self) -> list[int]:
        return [ self.kernel_rows, self.kernel_cols, self.kernel_depth ]

    @property
    def stride(self) -> list[int]:
        return [ self.stride_rows, self.stride_cols, self.stride_depth ]

    @property
    def pad(self) -> list[int]:
        return [
            self.pad_top,
            self.pad_left,
            self.pad_front,
            self.pad_bottom,
            self.pad_right,
            self.pad_back,
        ]

    @kernel_size.setter
    def kernel_size(self, val: list[int]) -> None:
        assert(len(val) == 3, "kernel size must be a list of three integers")
        self.kernel_rows    = val[0]
        self.kernel_cols    = val[1]
        self.kernel_depth   = val[2]

    @stride.setter
    def stride(self, val: list[int]) -> None:
        assert(len(val) == 3, "stride must be a list of three integers")
        self.stride_rows    = val[0]
        self.stride_cols    = val[1]
        self.stride_depth   = val[2]

    @pad.setter
    def pad(self, val: list[int]) -> None:
        assert(len(val) == 6, "pad must be a list of six integers")
        self.pad_top    = val[0]
        self.pad_right  = val[4]
        self.pad_bottom = val[3]
        self.pad_left   = val[1]
        self.pad_front  = val[2]
        self.pad_back   = val[5]

