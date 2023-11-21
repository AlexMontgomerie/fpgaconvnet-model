import importlib
import math
from typing import Union, ClassVar
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from collections import OrderedDict

import pydot
import numpy as np
from dacite import from_dict

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

from fpgaconvnet.models.layers import LayerBase, Layer2D, Layer3D
from fpgaconvnet.models.layers.traits import LayerMatchingCoarse
from fpgaconvnet.models.modules import ModuleBase

from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

@dataclass(kw_only=True)
class PoolingLayerBase(LayerMatchingCoarse, LayerBase):

    pool_type: str = 'max'
    kernel_rows: int = 2
    kernel_cols: int = 2
    stride_rows: int = 2
    stride_cols: int = 2
    pad_top: int = 0
    pad_right: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    data_t: FixedPoint = FixedPoint(16,8)

    name: ClassVar[str] = "pooling"

    def rows_out(self) -> int:
        return self.modules["sliding_window"].rows_out()

    def cols_out(self) -> int:
        return self.modules["sliding_window"].cols_out()

    @property
    def kernel_size(self) -> list[int]:
        return [ self.kernel_rows, self.kernel_cols ]

    @property
    def stride(self) -> list[int]:
        return [ self.stride_rows, self.stride_cols ]

    @property
    def pad(self) -> list[int]:
        return [
            self.pad_top,
            self.pad_bottom,
            self.pad_left,
            self.pad_right,
        ]

    @kernel_size.setter
    def kernel_size(self, val: list[int]) -> None:
        self.kernel_rows = val[0]
        self.kernel_cols = val[1]

    @stride.setter
    def stride(self, val: list[int]) -> None:
        self.stride_rows = val[0]
        self.stride_cols = val[1]

    @pad.setter
    def pad(self, val: list[int]) -> None:
        self.pad_top    = val[0]
        self.pad_right  = val[3]
        self.pad_bottom = val[2]
        self.pad_left   = val[1]

    @property
    def module_lookup(self) -> dict:

        match self.backend:
            case BACKEND.HLS:
                return OrderedDict({
                    "sliding_window": self.get_sliding_window_parameters,
                    "pool": self.get_pool_parameters,
                })

            case BACKEND.CHISEL:
                return OrderedDict({
                    "pad": self.get_pad_parameters,
                    "sliding_window": self.get_sliding_window_parameters,
                    "pool": self.get_pool_parameters,
                })

    def get_pad_parameters(self) -> dict:

        return {
            "repetitions": 1,
            "streams": self.streams(),
            "rows": self.rows,
            "cols": self.cols,
            "channels": self.channels//self.streams(),
            "pad_top": self.pad_top,
            "pad_right": self.pad_right,
            "pad_bottom": self.pad_bottom,
            "pad_left": self.pad_left,
            "data_t": self.data_t,
        }

    def get_sliding_window_parameters(self) -> dict:

        match self.backend:

            case BACKEND.HLS:

                return {
                    **self.input_shape_dict(),
                    "channels": self.channels//self.streams(),
                    "pad_top": self.pad_top,
                    "pad_right": self.pad_right,
                    "pad_bottom": self.pad_bottom,
                    "pad_left": self.pad_left,
                    "kernel_size": self.kernel_size,
                    "stride": self.stride,
                    "data_t": self.data_t,
                }

            case BACKEND.CHISEL:

                # get the dimensions from the pad module
                rows, cols, channels = self.modules["pad"].output_iter_space[0]

                return {
                    "repetitions": 1,
                    "streams": self.streams(),
                    "rows": rows,
                    "cols": cols,
                    "channels": channels//self.streams(),
                    "kernel_size": self.kernel_size,
                    "stride": self.stride,
                    "data_t": self.data_t,
                }

    def get_pool_parameters(self) -> dict:

        match self.backend:

            case BACKEND.HLS:
                return {
                    **self.output_shape_dict(),
                    "channels": self.channels//self.streams(),
                    "kernel_size": self.kernel_size,
                    "pool_type": self.pool_type,
                    "data_t": self.data_t,
                }

            case BACKEND.CHISEL:
                return {
                    "repetitions": int(np.prod(self.output_shape()))//self.streams(),
                    "streams": self.streams(),
                    "kernel_size": self.kernel_size,
                    "pool_type": self.pool_type,
                    "data_t": self.data_t,
                }

    def functional_model(self,data,batch_size=1):
        import torch

        assert data.shape == self.inputs_shape(), "ERROR: invalid input shape dimension"

        # instantiate pooling layer FIXME
        pooling_layer = torch.nn.MaxPool2d(self.kernel_size, stride=self.stride, padding=self.pad[0])

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return pooling_layer(torch.from_numpy(data)).detach().numpy()

class PoolingLayerBase3D(PoolingLayerBase):
    kernel_depth: int = 2
    stride_depth: int = 2
    pad_front: int = 0
    pad_back: int = 0

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
            self.pad_bottom,
            self.pad_left,
            self.pad_right,
            self.pad_front,
            self.pad_back,
        ]

    @kernel_size.setter
    def kernel_size(self, val: list[int]) -> None:
        self.kernel_rows = val[0]
        self.kernel_cols = val[1]
        self.kernel_depth = val[2]

    @stride.setter
    def stride(self, val: list[int]) -> None:
        self.stride_rows = val[0]
        self.stride_cols = val[1]
        self.stride_depth = val[2]

    @pad.setter
    def pad(self, val: list[int]) -> None:
        self.pad_top    = val[0]
        self.pad_bottom = val[1]
        self.pad_left   = val[2]
        self.pad_right  = val[3]
        self.pad_front  = val[4]
        self.pad_back   = val[5]

class PoolingLayerChisel(Layer2D, PoolingLayerBase):
    backend: ClassVar[BACKEND] = BACKEND.CHISEL
    register: ClassVar[bool] = True

class PoolingLayerHLS(PoolingLayerBase, Layer2D):
    backend: ClassVar[BACKEND] = BACKEND.HLS
    register: ClassVar[bool] = True

class PoolingLayerHLS3D(PoolingLayerBase3D, Layer3D):
    backend: ClassVar[BACKEND] = BACKEND.HLS
    register: ClassVar[bool] = True

