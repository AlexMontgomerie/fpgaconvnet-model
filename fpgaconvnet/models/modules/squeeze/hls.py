from typing import ClassVar, Optional
from dataclasses import dataclass

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleHLSBase, ModuleHLS3DBase, Port
from fpgaconvnet.models.modules.squeeze import lcm
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model
from fpgaconvnet.platform import DEFAULT_HLS_PLATFORM


@dataclass(kw_only=True)
class SqueezeHLSBase(ModuleHLSBase):

    # hardware parameters
    coarse_in: int
    coarse_out: int
    data_t: FixedPoint = FixedPoint(16, 8)

    # class variables
    name: ClassVar[str] = "squeeze"
    register: ClassVar[bool] = False

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.coarse_in],
            data_type=self.data_t,
            buffer_depth=2,
            name="io_in"
        )]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.coarse_out],
            data_type=self.data_t,
            buffer_depth=2,
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


@dataclass
class SqueezeHLS(SqueezeHLSBase):

    register: ClassVar[bool] = True

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.channels//self.coarse_in] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.channels//self.coarse_out] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.channels, self.coarse_in,
                self.coarse_out, lcm(self.coarse_in, self.coarse_out), self.data_t.width ]

@dataclass
class SqueezeHLS3D(ModuleHLS3DBase, SqueezeHLSBase):

    register: ClassVar[bool] = True

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.depth, self.channels//self.coarse_in] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.depth, self.channels//self.coarse_out] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.depth, self.channels, self.coarse_in,
                self.coarse_out, lcm(self.coarse_in, self.coarse_out), self.data_t.width ]

try:
    DEFAULT_SQUEEZE_RSC_MODELS: dict[str, ResourceModel] = { rsc_type: get_cached_resource_model(SqueezeHLS,
                                    rsc_type, "default") for rsc_type in DEFAULT_HLS_PLATFORM.resource_types }
except FileNotFoundError:
    print("CRITICAL WARNING: default resource models not found for Squeeze, default resource modelling will fail")

@eval_resource_model.register
def _(m: SqueezeHLS, rsc_type: str, _model: Optional[ResourceModel] = None) -> int:

    return 0 # TODO

    # # get the resource model
    # model: ResourceModel = _model if _model is not None else DEFAULT_SQUEEZE_RSC_MODELS[rsc_type]

    # # check the correct resource type
    # assert rsc_type == model.rsc_type, f"Incompatible resource type with model: {rsc_type}"

    # # get the resource model
    # match rsc_type:
    #     case "DSP":
    #         return 0
    #     case "BRAM":
    #         return 0
    #     case _:
    #         return model(m)

