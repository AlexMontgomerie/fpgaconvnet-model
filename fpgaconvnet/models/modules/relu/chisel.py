from typing import ClassVar, Optional
from dataclasses import dataclass, field

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model
from fpgaconvnet.platform import DEFAULT_CHISEL_PLATFORM

@dataclass(kw_only=True)
class ReLUChisel(ModuleChiselBase):

    # hardware parameters
    data_t: FixedPoint = field(default_factory=lambda: FixedPoint(16, 8))
    input_buffer_depth: int = 2
    output_buffer_depth: int = 2

    # class variables
    name: ClassVar[str] = "relu"
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
        return [ self.streams, self.data_t.width,
                self.input_buffer_depth, self.output_buffer_depth ]

    # def rsc(self, coef=None, model=None):
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

    def functional_model(self, *inputs: np.ndarray) -> np.ndarray:

        # get the input data
        data = inputs[0]

        # check input dimensions
        iter_space_len = len(self.input_iter_space[0])
        assert(len(data.shape) >= iter_space_len)
        assert(list(data.shape[-iter_space_len:]) == self.input_iter_space[0])

        # maximum of 0 and the data
        return np.maximum(data, 0.0)

try:
    DEFAULT_RELU_RSC_MODELS: dict[str, ResourceModel] = { rsc_type: get_cached_resource_model(ReLUChisel,
                                    rsc_type, "default") for rsc_type in DEFAULT_CHISEL_PLATFORM.resource_types }
except FileNotFoundError:
    print("CRITICAL WARNING: default resource models not found for ReLU, default resource modelling will fail")

@eval_resource_model.register
def _(m: ReLUChisel, rsc_type: str, _model: Optional[ResourceModel] = None) -> int:

    # get the resource model
    model: ResourceModel = _model if _model is not None else DEFAULT_RELU_RSC_MODELS[rsc_type]

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

