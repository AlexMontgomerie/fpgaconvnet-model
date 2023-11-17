import random
from typing import ClassVar, List
from dataclasses import dataclass

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import ModuleBase, Port


@dataclass(kw_only=True)
class AccumChisel(ModuleBase):

    # hardware parameters
    channels: int
    filters: int
    data_t: FixedPoint = FixedPoint(32, 16)
    streams: int = 1
    ram_style: str = "distributed"
    input_buffer_depth: int = 0
    acc_buffer_depth: int = 3
    output_buffer_depth: int = 0

    # class variables
    name: ClassVar[str] = "accum"
    backend: ClassVar[str] = "chisel"
    dimensionality: ClassVar[int] = 2

    # port information
    ports_in: ClassVar[int] = 1
    ports_out: ClassVar[int] = 1

    """
    Class Methods and Properties
    """

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            iteration_space=[self.channels, self.filters, self.streams],
            simd_lanes=[self.streams],
            data_type=self.data_t,
            buffer_depth=self.input_buffer_depth,
            name="io_in"
        )]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            iteration_space=[self.filters, self.streams],
            simd_lanes=[self.streams],
            data_type=self.data_t,
            buffer_depth=self.output_buffer_depth,
            name="io_out"
        )]

    def rate_in(self, idx: int) -> float:
        assert(idx == 0)
        return 1.0

    def rate_out(self, idx: int) -> float:
        assert(idx == 0)
        return 1.0/float(self.channels)

    def functional_model(self, data: np.ndarray) -> np.ndarray:

        # check input dimensions
        iter_space_len = len(self.input_iteration_space(0))
        assert(len(data.shape) >= iter_space_len)
        assert(data.shape[-iter_space_len] == self.input_iteration_space(0))

        # accumulate across the channel dimension
        return np.sum(data, axis=-3)

    def resource_parameters(self) -> list[int]:
        ram_style_int = 0 if self.ram_style == "distributed" else 1 # TODO: use an enumeration instead
        return [self.channels, self.filters, self.streams, self.data_t.width, ram_style_int,
                self.input_buffer_depth, self.acc_buffer_depth, self.output_buffer_depth]

    @classmethod
    def generate_random_configuration(cls):

        # generate a random configuration
        data_width = random.choice([4, 8, 16, 32])
        config = {
            "repetitions": random.randint(1, 64),
            "channels": random.randint(1, 512),
            "filters": random.randint(1, 512),
            "streams": random.randint(1, 64),
            "data_t": {
                "width": data_width,
                "binary_point": random.randint(0, data_width-1)
            },
            "ram_style": random.choice(["distributed", "block"]),
            "activity": {
                "input": random.uniform(0, 1.0),
            },
            "description": "Randomly generated Accum configuration"
        }

        # return the configuration
        return config

