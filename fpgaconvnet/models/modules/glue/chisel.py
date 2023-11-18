import random
from typing import ClassVar, List
from dataclasses import dataclass

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port

@dataclass(kw_only=True)
class GlueChisel(ModuleChiselBase):

    # hardware parameters
    coarse: int
    data_t: FixedPoint = FixedPoint(32, 16)
    input_buffer_depth: int = 0
    output_buffer_depth: int = 0

    # class variables
    name: ClassVar[str] = "glue"
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
            simd_lanes=[self.streams, self.coarse],
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
        return [ 1.0 ]

    @property
    def rate_out(self) -> list[float]:
        return [ 1.0 ]

    def pipeline_depth(self) -> int:
        return 1

    def resource_parameters(self) -> list[int]:
        return [ self.coarse, self.streams, self.data_t.width,
                self.input_buffer_depth, self.output_buffer_depth ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
            return {
                "Logic_LUT" : np.array([
                    self.streams*self.data_width*self.coarse_in, # tree buffer
                    self.streams*self.data_width*int2bits(self.coarse_in), # tree buffer
                    self.coarse_in, # input ready
                    1,
                ]),
                "LUT_RAM" : np.array([
                    # queue_lutram_resource_model(
                    #     int2bits(self.coarse_in)+1, self.streams*self.data_width), # buffer
                    1,
                ]),
                "LUT_SR" : np.array([
                    int2bits(self.coarse_in), # tree buffer valid
                    1,
                ]),
                "FF" : np.array([
                    self.coarse_in, # coarse in parameter
                    self.streams*self.data_width, # output buffer
                    int2bits(self.coarse_in), # tree buffer valid
                    self.streams*self.data_width*(2**(int2bits(self.coarse_in))), # tree buffer registers
                    self.streams*self.data_width*self.coarse_in, # tree buffer registers
                    1,
                ]),
                "DSP"       : np.array([0]),
                "BRAM36"    : np.array([0]),
                "BRAM18"    : np.array([0]),
            }


    def functional_model(self, data):

        # check input dimensions
        iter_space_len = len(self.input_iter_space[0])
        assert(len(data.shape) >= iter_space_len)
        assert(data.shape[-iter_space_len] == self.input_iter_space[0])

        # accumulate the data in the coarse dimension
        return np.sum(data, axis=-1)

