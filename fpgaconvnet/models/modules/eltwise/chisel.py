import random
from typing import ClassVar, List
from dataclasses import dataclass, field

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY

@dataclass(kw_only=True)
class EltwiseChisel(ModuleChiselBase):

    # hardware parameters
    ports: int
    data_t: FixedPoint = FixedPoint(16, 8)
    eltwise_type: str = "add"
    input_buffer_depth: list[int] = field(default_factory=list)
    output_buffer_depth: int = 0

    # class variables
    name: ClassVar[str] = "eltwise"
    register: ClassVar[bool] = True

    def __post_init__(self):
        if len(self.input_buffer_depth) == 0:
            self.input_buffer_depth = [0]*self.ports

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.streams],
            data_type=self.data_t,
            buffer_depth=self.input_buffer_depth[i],
            name=f"io_in_{i}"
        ) for i in range(self.ports) ]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.streams],
            data_type=self.data_t,
            buffer_depth=self.output_buffer_depth,
            name="io_out"
        )]

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [1] for i in range(self.ports) ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [1] ]

    @property
    def rate_in(self) -> list[float]:
        return [ 1.0 for i in range(self.ports) ]

    @property
    def rate_out(self) -> list[float]:
        return [ 1.0 ]

    def pipeline_depth(self) -> int:
        return 1

    def resource_parameters(self) -> list[int]:
        return [ self.streams, self.data_t.width,
                sum(self.input_buffer_depth), self.output_buffer_depth ]

    def functional_model(self, data):
        # check input dimensionality
        assert len(data) == self.ports , "ERROR: invalid number of ports"

        # perform elment wise operation
        match self.eltwise_type:
            case "add":
                return np.sum(data, axis=0)
            case "mul":
                return np.prod(data, axis=0)
            case _:
                raise ValueError(f"Element-wise type {self.eltwise_type} not supported")

