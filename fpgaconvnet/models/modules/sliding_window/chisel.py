import math
from typing import ClassVar, Optional
from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model
from fpgaconvnet.architecture import DIMENSIONALITY

@dataclass(kw_only=True)
class SlidingWindowChisel(ModuleChiselBase):

    # hardware parameters
    rows: int
    cols: int
    channels: int
    kernel_size: list[int]
    stride: list[int]
    data_t: FixedPoint = field(default_factory=lambda: FixedPoint(16, 8))
    window_buffer_ram_style: str = "distributed"
    line_buffer_ram_style: str = "block"
    input_buffer_depth: int = 0
    output_buffer_depth: int = 0

    # class variables
    name: ClassVar[str] = "sliding_window"
    dimensionality: ClassVar[set[DIMENSIONALITY]] = { DIMENSIONALITY.TWO }
    register: ClassVar[bool] = True

    def __post_init__(self):
        assert len(self.kernel_size) == 2, "SlidingWindowChisel only supports 2D kernel size"
        assert len(self.stride) == 2, "SlidingWindowChisel only supports 2D stride"

    @property
    def rows_out(self) -> int:
        return math.ceil((self.rows-self.kernel_size[0]+1)/self.stride[0])

    @property
    def cols_out(self) -> int:
        return math.ceil((self.cols-self.kernel_size[1]+1)/self.stride[1])

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.channels] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows_out, self.cols_out, self.channels] ]

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
            simd_lanes=[self.streams, *self.kernel_size],
            data_type=self.data_t,
            buffer_depth=self.output_buffer_depth,
            name="io_out"
        )]

    @property
    def rate_in(self) -> list[float]:
        return [ 1.0 ]

    @property
    def rate_out(self) -> list[float]:
        return [ (self.rows_out*self.cols_out)/float(self.rows*self.cols) ]

    def pipeline_depth(self) -> int:
        return (self.kernel_size[0]-1)*self.cols*self.channels + (self.kernel_size[1]-1)*self.channels

    def resource_parameters(self) -> list[int]:
        window_buffer_ram_style_int = 0 if self.window_buffer_ram_style == "distributed" else 1 # TODO: use an enumeration instead
        line_buffer_ram_style_int = 0 if self.line_buffer_ram_style == "distributed" else 1 # TODO: use an enumeration instead
        return [ self.rows, self.cols, self.channels,
                self.streams, self.data_t.width,
                window_buffer_ram_style_int, line_buffer_ram_style_int,
                self.input_buffer_depth, self.output_buffer_depth,
                *self.kernel_size, *self.stride ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
            return {
                "Logic_LUT" : [
                    self.data_t.width,
                    (self.kernel_size[0]-1),
                    self.kernel_size[0]*(self.kernel_size[1]-1),
                    self.data_t.width*(self.kernel_size[0]-1),
                    self.data_t.width*self.kernel_size[0]*(self.kernel_size[1]-1),
                ],
                "LUT_RAM"   : [1
                    # line_buffer_lutram, # line buffer
                    # window_buffer_lutram, # window buffer
                    # frame_buffer_lutram, # frame buffer
                ],
                "LUT_SR"    : [1],
                "FF"        : [
                    int2bits(self.rows), # row_cntr
                    int2bits(self.cols), # col_cntr
                    int2bits(self.channels), # channel_cntr
                    self.data_t.width, # input buffer
                    self.data_t.width*self.kernel_size[0]*self.kernel_size[1], # output buffer (data)
                    self.kernel_size[0]*self.kernel_size[1], # output buffer (valid)
                    self.data_t.width*(self.kernel_size[0]-1), # line buffer
                    self.data_t.width*self.kernel_size[0]*(self.kernel_size[1]-1), # window buffer
                    self.kernel_size[0]*self.kernel_size[1], # frame buffer
                    1,
                ],
                "DSP"       : [0],
                "BRAM36"    : [0],
                "BRAM18"    : [0],
            }

    # def rsc(self,coef=None, model=None):

    #     # get the linear model estimation
    #     rsc = Module.rsc(self, coef, model)

    #     if self.regression_model == "linear_regression":
    #         # get the buffer estimates
    #         line_buffer_bram, _, window_buffer_bram, _, _ = self.buffer_estimate()

    #         # add the bram estimation
    #         rsc["BRAM"] = line_buffer_bram + window_buffer_bram

    #         # ensure zero DSPs
    #         rsc["DSP"] = 0

    #     # return the resource usage
    #     return rsc

    def functional_model(self, *inputs: np.ndarray) -> np.ndarray:

        # get the input data
        data = inputs[0]

        # check input dimensions
        iter_space_len = len(self.input_iter_space[0])
        assert(len(data.shape) >= iter_space_len)
        assert(list(data.shape[-iter_space_len:]) == self.input_iter_space[0])

        # generate the windows
        windows = sliding_window_view(data, self.kernel_size, axis=(-3, -2))

        # stride across the windows
        windows = windows[..., :: self.stride[0], :: self.stride[1], :]

        # return the windows
        return windows

@eval_resource_model.register
def _(m: SlidingWindowChisel, rsc_type: str, model: ResourceModel) -> int:

    # check the correct resource type
    assert rsc_type == model.rsc_type, f"Incompatible resource type with model: {rsc_type}"

    # get the resource model
    match rsc_type:
        case "DSP":
            return 0
        case _:
            return model(m)

