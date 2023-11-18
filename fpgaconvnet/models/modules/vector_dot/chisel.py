import random
from typing import ClassVar, List
from dataclasses import dataclass

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY

@dataclass(kw_only=True)
class VectorDotChisel(ModuleChiselBase):

    # hardware parameters
    fine: int
    filters: int
    data_t: FixedPoint = FixedPoint(16, 8)
    weight_t: FixedPoint = FixedPoint(16, 8)
    acc_t: FixedPoint = FixedPoint(32, 16)
    use_dsp: bool = True
    input_buffer_depth: int = 2
    weight_buffer_depth: int = 2
    output_buffer_depth: int = 2

    # class variables
    name: ClassVar[str] = "vector_dot"
    register: ClassVar[bool] = True

    @property
    def input_ports(self) -> list[Port]:
        return [
            Port(
                simd_lanes=[self.streams, self.fine],
                data_type=self.data_t,
                buffer_depth=self.input_buffer_depth,
                name="io_in"
            ),
            Port(
                simd_lanes=[self.streams, self.fine],
                data_type=self.weight_t,
                buffer_depth=self.weight_buffer_depth,
                name="io_weights"
            ),

        ]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.streams],
            data_type=self.acc_t,
            buffer_depth=self.output_buffer_depth,
            name="io_out"
        )]

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [1], [self.filters] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.filters] ]

    @property
    def rate_in(self) -> list[float]:
        return [ (1.0/float(self.filters)), 1.0 ]

    @property
    def rate_out(self) -> list[float]:
        return [ 1.0 ]

    # def pipeline_depth(self) -> int:
    #     return self.filters*(self.channels-1)

    def resource_parameters(self) -> list[int]:
        return [ self.filters, self.streams, self.fine, np.prod(self.kernel_size),
                self.input_buffer_depth, self.weight_buffer_depth, self.output_buffer_depth,
                self.data_t.width, self.weight_t.width, self.acc_t.width ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return {
            "Logic_LUT" : np.array([
                self.fine, self.data_t.width, self.weight_width,
                self.streams*self.data_t.width*self.fine,
                self.streams*self.weight_width*self.fine,
                self.streams*self.acc_width*self.fine, # adder tree
                self.filters, # ready logic
                int2bits(self.filters), # filter counter
                1,
            ]),
            "LUT_RAM"   : np.array([
                # queue_lutram_resource_model(
                #     int2bits(self.fine)+1, self.streams*self.acc_width), # buffer
                1,
            ]),
            "LUT_SR"    : np.array([
                int2bits(self.fine)+1, # tree buffer valid
            ]),
            "FF"    : np.array([
                self.acc_width, # output buffer TODO
                self.filters, self.fine, # parameters
                int2bits(self.filters), # filter counter
                int2bits(self.fine)+1, # tree buffer valid
                self.streams*self.acc_width*self.fine, # adder tree reg
                self.streams*self.acc_width, # output buffer
                1,
            ]),
            "DSP"       : np.array([self.streams*self.fine]),
            "BRAM36"    : np.array([0]),
            "BRAM18"    : np.array([0]),
        }

    def functional_model(self, data: np.ndarray, weights: np.ndarray) -> np.ndarray:

        # replicate for filter dimension
        partial = np.repeat(np.expand_dims(data, axis=-3), self.filters, axis=-3)

        # multiply weights and data
        partial = np.multiply(partial, weights)

        # sum across the kernel dimension
        return np.sum(partial, axis=-1)

