from typing import ClassVar
from dataclasses import dataclass

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleHLSBase, ModuleHLS3DBase, Port
from fpgaconvnet.models.modules.squeeze import lcm

@dataclass(kw_only=True)
class SqueezeHLSBase(ModuleHLSBase):

    # hardware parameters
    coarse_in: int
    coarse_out: int
    data_t: FixedPoint = FixedPoint(16, 8)

    # class variables
    name: ClassVar[str] = "squeeze"
    register: ClassVar[bool] = False

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.coarse_in],
            data_type=self.data_t,
            buffer_depth=2,
            name="io_in"
        )]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.coarse_out],
            data_type=self.data_t,
            buffer_depth=2,
            name="io_out"
        )]

    @property
    def rate_in(self) -> list[float]:
        return [ 1.0 ] # TODO

    @property
    def rate_out(self) -> list[float]:
        return [ 1.0 ] # TODO

    def pipeline_depth(self) -> int:
        return lcm(self.coarse_in, self.coarse_out)


@dataclass
class SqueezeHLS(SqueezeHLSBase):

    register: ClassVar[bool] = True

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.channels//self.coarse_in] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.channels//self.coarse_out] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.channels, self.coarse_in,
                self.coarse_out, lcm(self.coarse_in, self.coarse_out), self.data_t.width ]

@dataclass
class SqueezeHLS3D(ModuleHLS3DBase, SqueezeHLSBase):

    register: ClassVar[bool] = True

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.depth, self.channels//self.coarse_in] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.depth, self.channels//self.coarse_out] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.depth, self.channels, self.coarse_in,
                self.coarse_out, lcm(self.coarse_in, self.coarse_out), self.data_t.width ]

