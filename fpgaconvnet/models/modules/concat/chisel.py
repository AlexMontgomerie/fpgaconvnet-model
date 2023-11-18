import random
from typing import ClassVar, List
from dataclasses import dataclass

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY

# DEFAULT_FITTER = NNLSHeuristicResourceFitter()

@dataclass(kw_only=True)
class ConcatChisel(ModuleChiselBase):

    # hardware parameters
    ports: int
    channels: list[int]
    data_t: FixedPoint = FixedPoint(16, 8)
    input_buffer_depth: list[int] = []
    output_buffer_depth: int = 0

    # class variables
    name: ClassVar[str] = "concat"
    register: ClassVar[bool] = True

    def __post_init__(self):
        assert(len(self.channels) == self.ports)

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
        return [ [self.channels[i], self.streams] for i in range(self.ports) ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [sum(self.channels), self.streams] ]

    @property
    def rate_in(self) -> list[float]:
        return [ self.channels[i]/float(sum(self.channels)) for i in range(self.ports) ]

    @property
    def rate_out(self) -> list[float]:
        return [ 1.0 ]

    def pipeline_depth(self) -> int:
        return sum(self.channels)

    def resource_parameters(self) -> list[int]:
        return [ sum(self.channels), self.streams, self.data_t.width,
                self.input_buffer_depth, self.output_buffer_depth ]

    def functional_model(self, data: list[np.ndarray]) -> np.ndarray:

        # check input dimensionality
        assert len(data) == self.ports_in , "Not enough input ports"

        # concatenate along the channel dimension
        return np.concatenate(data, axis=-2)


