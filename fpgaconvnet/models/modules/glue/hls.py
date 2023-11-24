from typing import ClassVar
from dataclasses import dataclass

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleHLSBase, ModuleHLS3DBase, Port

@dataclass(kw_only=True)
class GlueHLSBase(ModuleHLSBase):

    # hardware parameters
    coarse_in: int
    coarse_out: int
    coarse_group: int
    filters: int
    data_t: FixedPoint = FixedPoint(16, 8)
    acc_t: FixedPoint = FixedPoint(32, 16)

    # class variables
    name: ClassVar[str] = "glue"
    register: ClassVar[bool] = False

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.coarse_in*self.coarse_out*self.coarse_group],
            data_type=self.acc_t,
            buffer_depth=0,
            name="io_in"
        )]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.coarse_out*self.coarse_group],
            data_type=self.data_t,
            buffer_depth=0,
            name="io_out"
        )]

    @property
    def rate_in(self) -> list[float]:
        return [ 1.0 ]

    @property
    def rate_out(self) -> list[float]:
        return [ 1.0 ]

    def pipeline_depth(self) -> int:
        return 1

    def functional_model(self, data):

        # TODO: check input dimensionality

        # accumulate the data in the coarse dimension
        return np.sum(data, axis=-3)

@dataclass
class GlueHLS(GlueHLSBase):

    register: ClassVar[bool] = True

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.filters] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.filters] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.filters, self.coarse_in,
                self.coarse_out, self.coarse_group, self.data_t.width, self.acc_t.width ]

@dataclass
class GlueHLS3D(ModuleHLS3DBase, GlueHLSBase):

    register: ClassVar[bool] = True

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.depth, self.filters] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.depth, self.filters] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.depth, self.filters, self.coarse_in,
                self.coarse_out, self.coarse_group, self.data_t.width, self.acc_t.width ]

