from typing import ClassVar, Optional
from dataclasses import dataclass

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model
from fpgaconvnet.platform import DEFAULT_CHISEL_PLATFORM

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

try:
    DEFAULT_FORK_RSC_MODELS: dict[str, ResourceModel] = { rsc_type: get_cached_resource_model(ForkChisel,
                                    rsc_type, "default") for rsc_type in DEFAULT_CHISEL_PLATFORM.resource_types }
except FileNotFoundError:
    print("CRITICAL WARNING: default resource models not found for Fork, default resource modelling will fail")

@eval_resource_model.register
def _(m: ForkChisel, rsc_type: str, _model: Optional[ResourceModel] = None) -> int:

    # get the resource model
    model: ResourceModel = _model if _model is not None else DEFAULT_FORK_RSC_MODELS[rsc_type]

    # check the correct resource type
    assert rsc_type == model.rsc_type, f"Incompatible resource type with model: {rsc_type}"

    # get the resource model
    match rsc_type:
        case "DSP":
            return 0
        case _:
            return model(m)

