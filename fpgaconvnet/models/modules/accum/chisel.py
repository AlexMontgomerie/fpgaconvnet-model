import random
from typing import ClassVar, List
from dataclasses import dataclass

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY

# DEFAULT_FITTER = NNLSHeuristicResourceFitter()

@dataclass(kw_only=True)
class AccumChisel(ModuleChiselBase):

    # hardware parameters
    channels: int
    filters: int
    data_t: FixedPoint = FixedPoint(32, 16)
    ram_style: str = "distributed"
    input_buffer_depth: int = 0
    acc_buffer_depth: int = 3
    output_buffer_depth: int = 0

    # class variables
    name: ClassVar[str] = "accum"
    register: ClassVar[bool] = True

    # def memory_usage(self):
    #     return int(self.filters/self.groups)*self.data_width

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
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.channels, self.filters] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.filters] ]

    @property
    def rate_in(self) -> list[float]:
        return [ 1.0 ]

    @property
    def rate_out(self) -> list[float]:
        return [ 1.0/float(self.channels) ]

    def pipeline_depth(self) -> int:
        return self.filters*(self.channels-1)

    def resource_parameters(self) -> list[int]:
        ram_style_int = 0 if self.ram_style == "distributed" else 1 # TODO: use an enumeration instead
        return [ self.channels, self.filters, self.streams, self.data_t.width, ram_style_int,
                self.input_buffer_depth, self.acc_buffer_depth, self.output_buffer_depth ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return {
                "Logic_LUT" : np.array([
                    self.filters, self.channels, # parameter logic
                    self.streams*self.data_t.width, # input word logic
                    self.streams, # input streams logic
                    int2bits(self.channels), # channel cntr
                    int2bits(self.filters), # filter cntr
                    1, # extra
                ]),
                "LUT_RAM"   : np.array([
                    # queue_lutram_resource_model(
                    #     2, self.streams*self.data_width), # output buffer
                    self.streams*self.data_t.width*self.filters, # filter memory memory (size)
                    self.streams*self.data_t.width, # filter memory memory (word width)
                    self.filters, # filter memory memory (depth)
                ]),
                "LUT_SR"    : np.array([0]),
                "ff"        : np.array([
                    self.data_t.width,  # input val cache
                    self.streams*self.data_t.width,  # input val cache
                    int2bits(self.channels), # channel_cntr
                    int2bits(self.filters), # filter cntr
                    self.channels, # channel parameter reg
                    self.filters, # filter parameter reg
                    1, # other registers
                ]),
                "DSP"       : np.array([0]),
                "BRAM36"    : np.array([0]),
                "BRAM18"    : np.array([0]),
            }


    def functional_model(self, data: np.ndarray) -> np.ndarray:

        # check input dimensions
        iter_space_len = len(self.input_iteration_space(0))
        assert(len(data.shape) >= iter_space_len)
        assert(data.shape[-iter_space_len] == self.input_iteration_space(0))

        # accumulate across the channel dimension
        return np.sum(data, axis=-3)


