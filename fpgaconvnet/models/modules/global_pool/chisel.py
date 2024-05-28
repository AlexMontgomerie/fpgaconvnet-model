from typing import ClassVar, Optional
from dataclasses import dataclass, field

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model

@dataclass(kw_only=True)
class GlobalPoolChisel(ModuleChiselBase):

    # hardware parameters
    rows: int
    cols: int
    channels: int
    data_t: FixedPoint = field(default_factory=lambda: FixedPoint(16, 8))
    acc_t: FixedPoint = field(default_factory=lambda: FixedPoint(32, 16))
    divisor_resolution: int = 32
    input_buffer_depth: int = 0
    output_buffer_depth: int = 0

    # class variables
    name: ClassVar[str] = "global_pool"
    register: ClassVar[bool] = True

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.channels] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.channels] ]

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
        return [ 1.0 ]

    @property
    def rate_out(self) -> list[float]:
        return [ 1.0/float(self.rows*self.cols) ]

    def pipeline_depth(self) -> int:
        return 1

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.channels, self.streams, self.data_t.width,
                self.divisor_resolution, self.input_buffer_depth, self.output_buffer_depth ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return super().resource_parameters_heuristics({
            "Logic_LUT" : [
                self.acc_t.width, # adder
                self.data_t.width, # adder
                int2bits(self.channels), # channel_cntr
                int2bits(self.rows*self.cols), # spatial cntr
                1,
            ],
            "LUT_RAM"   : [
                # queue_lutram_resource_model(
                #     4, self.data_t.width), # buffer
            ],
            "LUT_SR"    : [0],
            "FF"        : [
                self.data_t.width, # input cache
                int2bits(self.channels), # channel_cntr
                int2bits(self.rows*self.cols), # spatial cntr
                self.acc_t.width*self.channels, # accumulation reg
                1, # other registers
            ],
            "DSP"       : [0],
            "BRAM36"    : [0],
            "BRAM18"    : [0],
        })


    def functional_model(self, *inputs: np.ndarray) -> np.ndarray:

        # get the input data
        data = inputs[0]

        # check input dimensions
        data_iter_space_len = len(self.input_iter_space[0]) + len(self.input_simd_lanes[0])
        data_iter_space = [*self.input_iter_space[0], *self.input_simd_lanes[0]]
        assert(len(data.shape) >= data_iter_space_len), \
                f"{len(data.shape)} is not greater than or equal to {data_iter_space_len}"
        assert(list(data.shape[-data_iter_space_len:]) == data_iter_space), \
                f"{list(data.shape[-data_iter_space_len:])} is not equal to {data_iter_space}"

        # return average
        return np.average(data, axis=(-4,-3))


@eval_resource_model.register
def _(m: GlobalPoolChisel, rsc_type: str, model: ResourceModel) -> int:

    # check the correct resource type
    assert rsc_type == model.rsc_type, f"Incompatible resource type with model: {rsc_type}"

    # get the resource model
    return model(m)

