from typing import ClassVar, Optional
from dataclasses import dataclass

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port, CHISEL_RSC_TYPES
from fpgaconvnet.models.modules.squeeze import lcm
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model
from fpgaconvnet.platform import DEFAULT_CHISEL_PLATFORM


@dataclass(kw_only=True)
class SqueezeChisel(ModuleChiselBase):

    # hardware parameters
    coarse_in: int
    coarse_out: int
    data_t: FixedPoint = FixedPoint(16, 8)
    input_buffer_depth: int = 0
    output_buffer_depth: int = 0

    # class variables
    name: ClassVar[str] = "squeeze"
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
        return [ 1.0 ] # TODO

    @property
    def rate_out(self) -> list[float]:
        return [ 1.0 ] # TODO

    def pipeline_depth(self) -> int:
        return lcm(self.coarse_in, self.coarse_out)

    def resource_parameters(self) -> list[int]:
        return [ self.coarse_in, self.coarse_out,
                lcm(self.coarse_in, self.coarse_out),
                self.streams, self.data_t.width,
                self.input_buffer_depth, self.output_buffer_depth ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        buffer_size = lcm(self.coarse_in, self.coarse_out)
        return {
            "Logic_LUT" : [
                (buffer_size//self.coarse_in), # buffer ready
                self.data_t.width*self.coarse_out*(buffer_size//self.coarse_out), # arbiter logic
                (buffer_size//self.coarse_in),
                (buffer_size//self.coarse_out),
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
                int2bits(buffer_size//self.coarse_in), # cntr_in
                buffer_size, # buffer registers
                self.data_t.width*self.coarse_out, # DCFull on the output (data)
                self.coarse_out, # DCFull on the output (ready and valid)
                self.coarse_out*int2bits(buffer_size//self.coarse_out), # arbiter registers
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

try:
    DEFAULT_SQUEEZE_RSC_MODELS: dict[str, ResourceModel] = { rsc_type: get_cached_resource_model(SqueezeChisel,
                                    rsc_type, "default") for rsc_type in DEFAULT_CHISEL_PLATFORM.resource_types }
except FileNotFoundError:
    print("CRITICAL WARNING: default resource models not found for Squeeze, default resource modelling will fail")

@eval_resource_model.register
def _(m: SqueezeChisel, rsc_type: str, _model: Optional[ResourceModel] = None) -> int:

    # get the resource model
    model: ResourceModel = _model if _model is not None else DEFAULT_SQUEEZE_RSC_MODELS[rsc_type]

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

