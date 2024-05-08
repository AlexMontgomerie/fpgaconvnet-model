from typing import ClassVar, Optional
from dataclasses import dataclass, field

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port, CHISEL_RSC_TYPES
from fpgaconvnet.models.modules.squeeze import lcm
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model


@dataclass(kw_only=True)
class SqueezeChisel(ModuleChiselBase):

    # hardware parameters
    coarse_in: int
    coarse_out: int
    data_t: FixedPoint = field(default_factory=lambda: FixedPoint(16, 8))
    input_buffer_depth: int = 0
    output_buffer_depth: int = 0

    # class variables
    name: ClassVar[str] = "squeeze"
    register: ClassVar[bool] = True

    @property
    def buffer_size(self) -> int:
        return lcm(self.coarse_in, self.coarse_out)

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [1] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [1] ]

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.streams, self.coarse_in],
            data_type=self.data_t,
            buffer_depth=self.input_buffer_depth,
            name="io_in"
        )]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.streams, self.coarse_out],
            data_type=self.data_t,
            buffer_depth=self.output_buffer_depth,
            name="io_out"
        )]

    @property
    def rate_in(self) -> list[float]:
        return [ self.coarse_in / float(max(self.coarse_in, self.coarse_out)) ]

    @property
    def rate_out(self) -> list[float]:
        return [ self.coarse_out / float(max(self.coarse_in, self.coarse_out)) ]

    def pipeline_depth(self) -> int:
        return self.buffer_size

    def resource_parameters(self) -> list[int]:
        return [ self.coarse_in, self.coarse_out,
                self.buffer_size, self.streams, self.data_t.width,
                self.input_buffer_depth, self.output_buffer_depth ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return {
            "Logic_LUT" : [
                (self.buffer_size//self.coarse_in), # buffer ready
                self.data_t.width*self.coarse_out*(self.buffer_size//self.coarse_out), # arbiter logic
                (self.buffer_size//self.coarse_in),
                (self.buffer_size//self.coarse_out),
                self.coarse_in,
                self.coarse_out,
                self.data_t.width*self.coarse_out, # DCFull on the output
                1,
            ],
            "LUT_RAM"   : [
                # buffer_lutram, # buffer
                1,
            ],
            "LUT_SR"    : [0],
            "FF"        : [
                int2bits(self.buffer_size//self.coarse_in), # cntr_in
                self.buffer_size, # buffer registers
                self.data_t.width*self.coarse_out, # DCFull on the output (data)
                self.coarse_out, # DCFull on the output (ready and valid)
                self.coarse_out*int2bits(self.buffer_size//self.coarse_out), # arbiter registers
                1,

            ],
            "DSP"       : [0],
            "BRAM36"    : [0],
            "BRAM18"    : [0],
        }


    def functional_model(self, *inputs: np.ndarray) -> np.ndarray:

        # get the input data
        data = inputs[0]

        # check input dimensions
        iter_space_len = len(self.input_iter_space[0])
        assert(len(data.shape) >= iter_space_len)
        assert(list(data.shape[-iter_space_len:]) == self.input_iter_space[0])

        # add the bias term to the data
        return data


@eval_resource_model.register
def _(m: SqueezeChisel, rsc_type: str, model: ResourceModel) -> int:

    # check the correct resource type
    assert rsc_type == model.rsc_type, f"Incompatible resource type with model: {rsc_type}"

    # get the resource model
    match rsc_type:
        case "DSP":
            return 0
        case "BRAM":
            return 0
        case _:
            return model(m)

