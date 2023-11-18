import random
from typing import ClassVar, List
from dataclasses import dataclass

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port

@dataclass(kw_only=True)
class HardswishChisel(ModuleChiselBase):

    # hardware parameters
    input_t: FixedPoint = FixedPoint(16, 8)
    output_t: FixedPoint = FixedPoint(16, 8)
    input_buffer_depth: int = 0
    output_buffer_depth: int = 0

    # class variables
    name: ClassVar[str] = "hardswish"
    register: ClassVar[bool] = True

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [1] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [1] ]

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.streams],
            data_type=self.input_t,
            buffer_depth=self.input_buffer_depth,
            name="io_in"
        )]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.streams],
            data_type=self.output_t,
            buffer_depth=self.output_buffer_depth,
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

    def resource_parameters(self) -> list[int]:
        return [ self.streams, self.data_t.width,
                self.input_buffer_depth, self.output_buffer_depth ]

    @staticmethod
    def hardswish(x):
        return x * max(0, min(1, (x + 3) / 6))

    def functional_model(self, data):

        # apply hardswish to all elements
        return np.vectorize(self.hardswish)(data)

