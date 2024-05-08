from dataclasses import dataclass, field
from typing import ClassVar, Optional
import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import Port, ModuleBaseMeta, ModuleHLSBase, ModuleHLS3DBase, int2bits
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model


@dataclass
class ConvHLSBase(ModuleHLSBase):

    # hardware parameters
    fine: int
    filters: int
    kernel_size: list[int]
    groups: int = 1
    data_t: FixedPoint = field(default_factory=lambda: FixedPoint(16, 8))
    weight_t: FixedPoint = field(default_factory=lambda: FixedPoint(16, 8))
    acc_t: FixedPoint = field(default_factory=lambda: FixedPoint(32, 16))

    # class variables
    name: ClassVar[str] = "conv"
    register: ClassVar[bool] = False

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[*self.kernel_size],
            data_type=self.data_t,
            buffer_depth=0,
            name="in"
        )]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[1],
            data_type=self.acc_t,
            buffer_depth=0,
            name="out"
        )]

    @property
    def rate_in(self) -> list[float]:
        return [ self.fine*self.groups / float(np.prod(self.kernel_size)*self.filters) ]

    @property
    def rate_out(self) -> list[float]:
        return [ self.fine / float(np.prod(self.kernel_size)) ]

    def pipeline_depth(self):
        return self.fine

@dataclass
class ConvHLS(ConvHLSBase):

    register: ClassVar[bool] = True

    def __post_init__(self):

        # format kernel size as a 2 element list
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size]*2
        elif isinstance(self.kernel_size, list):
            assert len(self.kernel_size) == 2, "Must specify two kernel dimensions"
        else:
            raise TypeError

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.channels] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.channels, self.filters] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.groups, self.channels,
                self.filters, int(np.prod(self.kernel_size)), self.fine,
                self.data_t.width, self.weight_t.width, self.acc_t.width ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return {
            "LUT"  : [
                int2bits(self.filters),
                int2bits(self.cols*self.rows),
                int2bits(self.channels)
            ],
            "FF"   : [
                int2bits(self.filters),
                int2bits(self.cols*self.rows),
                int2bits(self.channels)
            ],
            "DSP"  : [1],
            "BRAM" : [1]
        }

    # def rsc(self,coef=None, model=None):
    #     # use module resource coefficients if none are given
    #     if coef == None:
    #         coef = self.rsc_coef
    #     # get an estimate for the dsp usage
    #     dot_product_dsp = self.fine * dsp_multiplier_resource_model(self.weight_width, self.data_width)
    #     # get the linear model estimation
    #     rsc = Module.rsc(self, coef, model)
    #     # update the dsp usage
    #     rsc["DSP"] = dot_product_dsp
    #     # set the BRAM usage to zero
    #     rsc["BRAM"] = 0
    #     # return the resource model
    #     return rsc

    def functional_model(self,data,weights):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"
        assert data.shape[3] == self.kernel_size[0]  , "ERROR: invalid column dimension"
        assert data.shape[4] == self.kernel_size[1]  , "ERROR: invalid column dimension"
        # check weight dimensionality
        assert weights.shape[0] == self.channels, "ERROR: invalid channel dimension"
        assert weights.shape[1] == int(self.filters/float(self.groups)) , "ERROR: invalid filter dimension"
        assert weights.shape[2] == self.kernel_size[0]  , "ERROR: invalid column dimension"
        assert weights.shape[3] == self.kernel_size[1]  , "ERROR: invalid column dimension"

        out = np.zeros((
            self.rows,
            self.cols,
            self.channels,
            int(self.filters/self.groups)
        ),dtype=float)

        for index,_ in np.ndenumerate(out):
            for k1 in range(self.kernel_size[0]):
                for k2 in range(self.kernel_size[1]):
                    out[index] += data[
                      index[0],index[1],index[2],k1,k2]*weights[
                      index[2],index[3],k1,k2]

        return out

@dataclass
class ConvHLS3D(ModuleHLS3DBase, ConvHLSBase):

    register: ClassVar[bool] = True

    def __post_init__(self):

        # format kernel size as a 3 element list
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size]*3
        elif isinstance(self.kernel_size, list):
            assert len(self.kernel_size) == 3, "Must specify three kernel dimensions"
        else:
            raise TypeError

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.depth, self.channels] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.depth, self.channels, self.filters] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.depth, self.groups, self.channels,
                self.filters, int(np.prod(self.kernel_size)), self.fine,
                self.data_t.width, self.weight_t.width, self.acc_t.width ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return {
            "LUT"  : [
                int2bits(self.filters),
                int2bits(self.cols*self.rows*self.depth),
                int2bits(self.channels)
            ],
            "FF"   : [
                int2bits(self.filters),
                int2bits(self.cols*self.rows*self.depth),
                int2bits(self.channels)
            ],
            "DSP"  : [1],
            "BRAM" : [1]
        }

    def functional_model(self,data,weights):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth    , "ERROR: invalid depth dimension"
        assert data.shape[3] == self.channels   , "ERROR: invalid channel dimension"
        assert data.shape[4] == self.kernel_rows  , "ERROR: invalid kernel row dimension"
        assert data.shape[5] == self.kernel_cols  , "ERROR: invalid kernel column dimension"
        assert data.shape[6] == self.kernel_depth  , "ERROR: invalid kernel depth dimension"
        # check weight dimensionality
        assert weights.shape[0] == self.channels, "ERROR: invalid channel dimension"
        assert weights.shape[1] == int(self.filters/float(self.groups)) , "ERROR: invalid filter dimension"
        assert weights.shape[2] == self.kernel_rows  , "ERROR: invalid kernel row dimension"
        assert weights.shape[3] == self.kernel_cols  , "ERROR: invalid kernel column dimension"
        assert weights.shape[4] == self.kernel_depth  , "ERROR: invalid kernel depth dimension"

        out = np.zeros((
            self.rows,
            self.cols,
            self.depth,
            self.channels,
            int(self.filters/self.groups)
        ),dtype=float)

        for index,_ in np.ndenumerate(out):
            for k1 in range(self.kernel_rows):
                for k2 in range(self.kernel_cols):
                    for k3 in range(self.kernel_depth):
                        out[index] += data[
                        index[0],index[1],index[2],index[3],k1,k2,k3]*weights[
                        index[2],index[3],index[4],k1,k2,k3]

        return out

@eval_resource_model.register
def _(m: ConvHLS, rsc_type: str, model: ResourceModel) -> int:

    # check the correct resource type
    assert rsc_type == model.rsc_type, f"Incompatible resource type with model: {rsc_type}"

    # get the resource model
    return model(m)

