from typing import ClassVar, Optional
from dataclasses import dataclass, field

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleHLSBase, ModuleHLS3DBase, Port
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model
from fpgaconvnet.platform import DEFAULT_HLS_PLATFORM

@dataclass(kw_only=True)
class ReLUHLSBase(ModuleHLSBase):

    # hardware parameters
    data_t: FixedPoint = field(default_factory=lambda: FixedPoint(16, 8))

    # class variables
    name: ClassVar[str] = "relu"
    register: ClassVar[bool] = False

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[1],
            data_type=self.data_t,
            buffer_depth=2,
            name="io_in"
        )]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[1],
            data_type=self.data_t,
            buffer_depth=2,
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

    def functional_model(self, data):

        # maximum of 0 and the data
        return np.maximum(data, 0.0)

@dataclass
class ReLUHLS(ReLUHLSBase):

    register: ClassVar[bool] = True

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.channels] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.channels] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.channels, self.data_t.width ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return {
            "LUT"   : [1],
            "FF"    : [1],
            "DSP"   : [0],
            "BRAM"  : [0],
        }

@dataclass
class ReLUHLS3D(ModuleHLS3DBase, ReLUHLSBase):

    register: ClassVar[bool] = True

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.depth, self.channels] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.depth, self.channels] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.depth, self.channels, self.data_t.width ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return {
            "LUT"   : [1],
            "FF"    : [1],
            "DSP"   : [0],
            "BRAM"  : [0],
        }

try:
    DEFAULT_RELU_RSC_MODELS: dict[str, ResourceModel] = { rsc_type: get_cached_resource_model(ReLUHLS,
                                    rsc_type, "default") for rsc_type in DEFAULT_HLS_PLATFORM.resource_types }
except FileNotFoundError:
    print("CRITICAL WARNING: default resource models not found for ReLU, default resource modelling will fail")

@eval_resource_model.register
def _(m: ReLUHLS, rsc_type: str, _model: Optional[ResourceModel] = None) -> int:

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

