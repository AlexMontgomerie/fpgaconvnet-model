from dataclasses import dataclass, field
from typing import ClassVar, Optional
import numpy as np

from fpgaconvnet.models.modules import Port, ModuleBaseMeta, ModuleHLSBase, ModuleHLS3DBase, int2bits
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model
from fpgaconvnet.data_types import FixedPoint
# from fpgaconvnet.platform import DEFAULT_HLS_PLATFORM

@dataclass
class AccumHLSBase(ModuleHLSBase):

    # hardware parameters
    filters: int
    groups: int
    accum_t: FixedPoint = field(default_factory=lambda: FixedPoint(32, 16))

    # class variables
    name: ClassVar[str] = "accum"
    register: ClassVar[bool] = False

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[1],
            data_type=self.accum_t,
            buffer_depth=0,
            name="in"
        )]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[1],
            data_type=self.accum_t,
            buffer_depth=0,
            name="out"
        )]

    @property
    def rate_in(self) -> list[float]:
        return [ 1.0 ]

    @property
    def rate_out(self) -> list[float]:
        return [ self.groups/float(self.channels) ]

    def pipeline_depth(self):
        return (self.channels*self.filters)//(self.groups*self.groups)

@dataclass
class AccumHLS(AccumHLSBase):

    register: ClassVar[bool] = True

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.channels, self.filters//self.groups] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.filters//self.groups] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.groups, self.channels, self.filters, self.accum_t.width ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return {
            "LUT" : [
                self.filters, self.groups, self.accum_t.width,
                self.cols, self.rows, self.channels
            ],
            "FF" : [
                self.filters, self.groups, self.accum_t.width,
                self.cols,self.rows,self.channels
            ],
            "DSP" : [
                self.filters, self.groups ,self.accum_t.width,
                self.cols, self.rows, self.channels
            ],
            "BRAM" : [
                self.filters, self.groups, self.accum_t.width,
                self.cols, self.rows, self.channels
            ],
        }

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows                   , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols                   , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels               , "ERROR: invalid channel dimension"
        assert data.shape[3] == self.filters//self.groups   , "ERROR: invalid filter  dimension"

        channels_per_group = self.channels//self.groups
        filters_per_group  = self.filters//self.groups

        out = np.zeros((
            self.rows,
            self.cols,
            self.filters),dtype=float)

        tmp = np.zeros((
            self.rows,
            self.cols,
            channels_per_group,
            filters_per_group),dtype=float)

        for index,_ in np.ndenumerate(tmp):
            for g in range(self.groups):
                out[index[0],index[1],g*filters_per_group+index[3]] = \
                        float(out[index[0],index[1],g*filters_per_group+index[3]]) + \
                        float(data[index[0],index[1],g*channels_per_group+index[2],index[3]])

        return out

@dataclass
class AccumHLS3D(ModuleHLS3DBase, AccumHLSBase):

    register: ClassVar[bool] = True

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.depth, self.channels, self.filters//self.groups] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.depth, self.filters//self.groups] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.depth, self.groups, self.channels, self.filters, self.accum_t.width ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return {
            "LUT" : [
                self.filters, self.groups, self.accum_t.width,
                self.cols, self.rows, self.channels, self.depth
            ],
            "FF" : [
                self.filters, self.groups, self.accum_t.width,
                self.cols, self.rows, self.channels, self.depth
            ],
            "DSP" : [
                self.filters, self.groups ,self.accum_t.width,
                self.cols, self.rows, self.channels, self.depth
            ],
            "BRAM" : [
                self.filters, self.groups, self.accum_t.width,
                self.cols, self.rows, self.channels, self.depth
            ],
        }

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows                   , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols                   , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth                  , "ERROR: invalid depth dimension"
        assert data.shape[3] == self.channels               , "ERROR: invalid channel dimension"
        assert data.shape[4] == self.filters//self.groups   , "ERROR: invalid filter  dimension"

        channels_per_group = self.channels//self.groups
        filters_per_group  = self.filters//self.groups

        out = np.zeros((
            self.rows,
            self.cols,
            self.depth,
            self.filters),dtype=float)

        tmp = np.zeros((
            self.rows,
            self.cols,
            self.depth,
            channels_per_group,
            filters_per_group),dtype=float)

        for index,_ in np.ndenumerate(tmp):
            for g in range(self.groups):
                out[index[0],index[1],index[2],g*filters_per_group+index[4]] = \
                        float(out[index[0],index[1],index[2],g*filters_per_group+index[4]]) + \
                        float(data[index[0],index[1],index[2],g*channels_per_group+index[3],index[4]])

        return out


# try:
#     DEFAULT_ACCUM_RSC_MODELS: dict[str, ResourceModel] = { rsc_type: get_cached_resource_model(AccumHLS,
#                                     rsc_type, "default") for rsc_type in DEFAULT_HLS_PLATFORM.resource_types }
# except FileNotFoundError:
#     print("CRITICAL WARNING: default resource models not found for Accum, default resource modelling will fail")

@eval_resource_model.register
def _(m: AccumHLS, rsc_type: str, _model: Optional[ResourceModel] = None) -> int:

    # get the resource model
    # model: ResourceModel = _model if _model is not None else DEFAULT_ACCUM_RSC_MODELS[rsc_type]
    model: ResourceModel = _model

    # check the correct resource type
    assert rsc_type == model.rsc_type, f"Incompatible resource type with model: {rsc_type}"

    # return zero if channels are 1
    if m.channels == 1:
        return 0

    # get the resource model
    match rsc_type:
        case "DSP":
            return 0
        case _:
            return model(m)

