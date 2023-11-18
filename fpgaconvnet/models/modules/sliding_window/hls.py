import random
from typing import ClassVar, Union
from dataclasses import dataclass

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleHLSBase, ModuleHLS3DBase, Port

@dataclass(kw_only=True)
class SlidingWindowHLSBase(ModuleHLSBase):

    # hardware parameters
    pad: Union[int,list[int]]
    stride: Union[list[int]]
    kernel_size: Union[list[int]]
    data_t: FixedPoint = FixedPoint(16, 8)

    # class variables
    name: ClassVar[str] = "sliding_window"
    register: ClassVar[bool] = False

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[1],
            data_type=self.data_t,
            buffer_depth=2,
            name="io_in"
        )]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[*self.kernel_size],
            data_type=self.data_t,
            buffer_depth=2,
            name="io_out"
        )]

@dataclass
class SlidingWindowHLS(SlidingWindowHLSBase): #FIXME

    register: ClassVar[bool] = True

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.channels] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows_out, self.cols_out, self.filters] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.channels, *self.pad,
                *self.stride, *self.kernel_size, self.data_t.width ]


@dataclass
class SlidingWindowHLS3D(ModuleHLS3DBase, SlidingWindowHLSBase): #FIXME

    register: ClassVar[bool] = True

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.depth, self.channels] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows_out, self.cols_out, self.depth_out, self.filters] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.depth, self.channels, *self.pad,
                *self.stride, *self.kernel_size, self.data_t.width ]

