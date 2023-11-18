import math
import random
from typing import ClassVar, Union
from dataclasses import dataclass

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port

# DEFAULT_FITTER = NNLSHeuristicResourceFitter()

@dataclass(kw_only=True)
class PoolChisel(ModuleChiselBase):

    # hardware parameters
    kernel_size: Union[list[int], int]
    data_t: FixedPoint = FixedPoint(16, 8)
    pool_type: str = "max"
    input_buffer_depth: int = 0
    output_buffer_depth: int = 0

    # class variables
    name: ClassVar[str] = "pool"
    register: ClassVar[bool] = True

    def __post_init__(self):

        # format kernel size as a 2 element list
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size]*2
        elif isinstance(self.kernel_size, list):
            assert len(self.kernel_size) == 2, "Must specify two kernel dimensions"
        else:
            raise TypeError

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [1] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [1] ]

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.streams,*self.kernel_size],
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
        return math.log(np.prod(self.kernel_size), 2)

    def resource_parameters(self) -> list[int]:
        return [ np.prod(self.kernel_size), self.streams, self.data_t.width,
                self.input_buffer_depth, self.output_buffer_depth ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return {
            "Logic_LUT"  : np.array([
                self.kernel_size[0]*self.kernel_size[1],
                self.data_width*self.kernel_size[0]*self.kernel_size[1], # tree buffer
                self.data_width*int2bits(self.kernel_size[0]*self.kernel_size[1]), # tree buffer
                self.kernel_size[0],self.kernel_size[1], # input ready
                1,
            ]),
            "LUT_RAM"  : np.array([
                # queue_lutram_resource_model(
                #     int2bits(self.kernel_size[0]*self.kernel_size[1])+1, self.data_width), # buffer
                1,
            ]),
            "LUT_SR"  : np.array([0]),
            "FF"   : np.array([
                self.data_width, # output buffer
                self.data_width*self.kernel_size[0]*self.kernel_size[1], # op tree input
                int2bits(self.kernel_size[0]*self.kernel_size[1]), # shift register
                1,
            ]),
            "DSP"  : np.array([0]),
            "BRAM36" : np.array([0]),
            "BRAM18" : np.array([0]),
        }


    def functional_model(self, data):

        # check input dimensions
        iter_space_len = len(self.input_iter_space[0])
        assert(len(data.shape) >= iter_space_len)
        assert(data.shape[-iter_space_len] == self.input_iter_space[0])

        # perform the pooling operation
        match self.pool_type:
            case 'max':
                return np.max(data, axis=-1)
            case 'avg':
                return np.mean(data, axis=-1)
            case _:
                raise ValueError(f"Invalid pool type: {self.pool_type}")


