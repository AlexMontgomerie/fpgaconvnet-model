import random
from typing import ClassVar, List
from dataclasses import dataclass

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port

@dataclass(kw_only=True)
class ResizeChisel(ModuleChiselBase):

    # hardware parameters
    rows: int
    cols: int
    channels: int
    scales: list[int]
    data_t: FixedPoint = FixedPoint(32, 16)
    ram_style: str = "distributed"
    input_buffer_depth: int = 0
    output_buffer_depth: int = 0

    # class variables
    name: ClassVar[str] = "resize"
    register: ClassVar[bool] = True

    def __post_init__(self):
        assert len(self.scales) == 3, "ResizeChisel only supports 3D scaling"

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.channels] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows*self.scales[0], self.cols*self.scales[1], self.channels*self.scales[2]] ]

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.streams],
            data_type=self.data_t,
            buffer_depth=self.input_buffer_depth,
            name="io_in"
        )]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.streams],
            data_type=self.data_t,
            buffer_depth=self.output_buffer_depth,
            name="io_out"
        )]

    @property
    def rate_in(self) -> list[float]:
        return [ 1.0 / np.prod(self.scales) ]

    @property
    def rate_out(self) -> list[float]:
        return [ 1.0 ]

    def pipeline_depth(self) -> int:
        return 1

    def resource_parameters(self) -> list[int]:
        ram_style_int = 0 if self.ram_style == "distributed" else 1 # TODO: use an enumeration instead
        return [ self.rows, self.cols, self.channels, self.streams, self.data_t.width,
                ram_style_int, self.input_buffer_depth, self.output_buffer_depth,
                *self.scales ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return {
            "Logic_LUT" : np.array([1]),
            "LUT_RAM"   : np.array([1]),
            "LUT_SR"    : np.array([0]),
            "FF"        : np.array([1]),
            "DSP"       : np.array([0]),
            "BRAM36"    : np.array([0]),
            "BRAM18"    : np.array([0]),
        }


    def functional_model(self, data):

        pass # TODO


