from typing import ClassVar, Optional
from dataclasses import dataclass

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model
from fpgaconvnet.platform import DEFAULT_CHISEL_PLATFORM


@dataclass(kw_only=True)
class HardswishChisel(ModuleChiselBase):

    # hardware parameters
    input_t: FixedPoint = FixedPoint(16, 8)
    output_t: FixedPoint = FixedPoint(16, 8)
    input_buffer_depth: int = 0
    output_buffer_depth: int = 0

    # class variables
    name: ClassVar[str] = "hardswish"
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
            data_type=self.input_t,
            buffer_depth=self.input_buffer_depth,
            name="io_in"
        )]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.streams],
            data_type=self.output_t,
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
        return [ self.streams, self.input_t.width, self.output_t.width,
                self.input_buffer_depth, self.output_buffer_depth ]

    @staticmethod
    def hardswish(x):
        return x * max(0, min(1, (x + 3) / 6))

    def functional_model(self, *inputs: np.ndarray) -> np.ndarray:

        # get the input data
        data = inputs[0]

        # check input dimensions
        iter_space_len = len(self.input_iter_space[0])
        assert(len(data.shape) >= iter_space_len)
        assert(list(data.shape[-iter_space_len:]) == self.input_iter_space[0])

        # apply hardswish to all elements
        return np.vectorize(self.hardswish)(data)


try:
    DEFAULT_GLUE_RSC_MODELS: dict[str, ResourceModel] = { rsc_type: get_cached_resource_model(HardswishChisel,
                                    rsc_type, "default") for rsc_type in DEFAULT_CHISEL_PLATFORM.resource_types }
except FileNotFoundError:
    print("CRITICAL WARNING: default resource models not found for Hardswish, default resource modelling will fail")

@eval_resource_model.register
def _(m: HardswishChisel, rsc_type: str, _model: Optional[ResourceModel] = None) -> int:

    # get the resource model
    model: ResourceModel = _model if _model is not None else DEFAULT_GLUE_RSC_MODELS[rsc_type]

    # check the correct resource type
    assert rsc_type == model.rsc_type, f"Incompatible resource type with model: {rsc_type}"

    # get the resource model
    return model(m)

