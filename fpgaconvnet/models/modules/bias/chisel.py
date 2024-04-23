from typing import ClassVar, Optional
from dataclasses import dataclass

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model
from fpgaconvnet.platform import DEFAULT_CHISEL_PLATFORM

@dataclass(kw_only=True)
class BiasChisel(ModuleChiselBase):

    # hardware parameters
    channels: int
    data_t: FixedPoint = FixedPoint(32, 16)
    ram_style: str = "distributed"
    input_buffer_depth: int = 2
    output_buffer_depth: int = 2

    # class variables
    name: ClassVar[str] = "bias"
    register: ClassVar[bool] = True

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.channels] ]

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
        return [ 1.0 ]

    def pipeline_depth(self) -> int:
        return 1

    def resource_parameters(self) -> list[int]:
        ram_style_int = 0 if self.ram_style == "distributed" else 1 # TODO: use an enumeration instead
        return [ self.channels, self.streams, self.data_t.width, ram_style_int,
                self.input_buffer_depth, self.output_buffer_depth ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return {
            "Logic_LUT" : [
                    self.data_t.width,
                ],
            "LUT_RAM"   : [
                    self.data_t.width*self.channels,
                ],
            "LUT_SR"    : [0],
            "FF"        : [
                    self.data_t.width,
                    int2bits(self.channels),
                    1,
                ],
            "DSP"       : [0],
            "BRAM36"    : [0],
            "BRAM18"    : [0],
        }


    def functional_model(self, *inputs: np.ndarray) -> np.ndarray:

        # get the input data
        data = inputs[0]
        biases = inputs[1]

        # check input dimensions
        iter_space_len = len(self.input_iter_space[0])
        assert(len(data.shape) >= iter_space_len)
        assert(list(data.shape[-iter_space_len:]) == self.input_iter_space[0])

        # add the bias term to the data
        return data + biases

try:
    DEFAULT_BIAS_RSC_MODELS: dict[str, ResourceModel] = { rsc_type: get_cached_resource_model(BiasChisel,
                                    rsc_type, "default") for rsc_type in DEFAULT_CHISEL_PLATFORM.resource_types }
except FileNotFoundError:
    print("CRITICAL WARNING: default resource models not found for Bias, default resource modelling will fail")

@eval_resource_model.register
def _(m: BiasChisel, rsc_type: str, _model: Optional[ResourceModel] = None) -> int:

    # TODO: fix me when there's a model
    return 0

    # # get the resource model
    # model: ResourceModel = _model if _model is not None else DEFAULT_BIAS_RSC_MODELS[rsc_type]

    # # check the correct resource type
    # assert rsc_type in CHISEL_RSC_TYPES, f"Invalid resource type: {rsc_type}"
    # assert rsc_type == model.rsc_type, f"Incompatible resource type with model: {rsc_type}"

    # # get the resource model
    # match rsc_type:
    #     case "DSP":
    #         return 0
    #     case _:
    #         return model(m)
