import random
from typing import ClassVar, List
from dataclasses import dataclass

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port

@dataclass(kw_only=True)
class PadChisel(ModuleChiselBase):

    # hardware parameters
    rows: int
    cols: int
    channels: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    pad_left: int
    pad_value: float = 0.0
    data_t: FixedPoint = FixedPoint(32, 16)
    input_buffer_depth: int = 2
    output_buffer_depth: int = 2

    # class variables
    name: ClassVar[str] = "pad"
    register: ClassVar[bool] = True

    @property
    def pad(self) -> list[int]:
        return [ self.pad_top, self.pad_right, self.pad_bottom, self.pad_left ]

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.channels] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows + self.pad_top + self.pad_bottom,
                  self.cols + self.pad_left + self.pad_right,
                  self.channels] ]

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
        return [ np.prod(self.input_iter_space[0]) / np.prod(self.output_iter_space[0]) ]

    @property
    def rate_out(self) -> list[float]:
        return [ 1.0 ]

    def pipeline_depth(self) -> int:
        return 1

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.channels, self.streams, self.data_t.width,
                self.pad_left, self.pad_right, self.pad_top, self.pad_bottom ]

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

    # def rsc(self):
    #     """
    #     Returns
    #     -------
    #     dict
    #         estimated resource usage of the module. Uses the
    #         resource coefficients for the estimate.
    #     """
    #     return {
    #         "LUT"   : 16,
    #         "FF"    : 35,
    #         "BRAM"  : 0,
    #         "DSP"   : 0
    #     }

    def functional_model(self, data):

        # get the output data from the functional model
        return np.pad(data, ((self.pad_top, self.pad_bottom),
            (self.pad_left, self.pad_right), (0,0)), 'constant')


