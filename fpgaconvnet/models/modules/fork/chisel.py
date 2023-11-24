from typing import ClassVar
from dataclasses import dataclass

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY

@dataclass(kw_only=True)
class ForkChisel(ModuleChiselBase):

    # hardware parameters
    fine: int
    coarse: int
    data_t: FixedPoint = FixedPoint(16, 8)
    is_sync: bool = False
    input_buffer_depth: int = 2
    output_buffer_depth: int = 2

    # class variables
    name: ClassVar[str] = "fork"
    register: ClassVar[bool] = True

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.streams, self.fine],
            data_type=self.data_t,
            buffer_depth=self.input_buffer_depth,
            name="io_in"
        )]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.streams, self.coarse, self.fine],
            data_type=self.data_t,
            buffer_depth=self.output_buffer_depth,
            name="io_out"
        )]

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [[1]]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [[1]]

    @property
    def rate_in(self) -> list[float]:
        return [ 1.0 ]

    @property
    def rate_out(self) -> list[float]:
        return [ 1.0 ]

    def pipeline_depth(self) -> int:
        return 1

    def resource_parameters(self) -> list[int]:
        return [ self.fine, self.coarse, self.streams, self.data_t.width,
                self.input_buffer_depth, self.output_buffer_depth ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return {
            "Logic_LUT" : [
                self.streams*self.fine, # input buffer
                self.streams*self.data_t.width*self.fine, # input buffer
                self.streams*self.fine*self.coarse, # output buffer
                self.streams*self.data_t.width*self.fine*self.coarse, # output buffer
                1,
            ],
            "LUT_RAM"   : [0],
            "LUT_SR"    : [0],
            "FF"    : [
                self.streams*self.fine, # input buffer (ready)
                self.streams*self.data_t.width*self.fine*self.coarse, # output buffer (data)
                1,
            ],
            "DSP"       : [0],
            "BRAM36"    : [0],
            "BRAM18"    : [0],
        }


    def functional_model(self, data):

        # replicate for coarse streams
        return np.repeat(np.expand_dims(data, axis=-2), self.coarse, axis=-2)

