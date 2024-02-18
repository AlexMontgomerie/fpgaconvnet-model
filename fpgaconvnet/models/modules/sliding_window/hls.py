import math
from typing import ClassVar, Optional
from dataclasses import dataclass

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleHLSBase, ModuleHLS3DBase, Port
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model
from fpgaconvnet.platform import DEFAULT_HLS_PLATFORM


@dataclass(kw_only=True)
class SlidingWindowHLSBase(ModuleHLSBase):

    # hardware parameters
    pad: list[int]
    stride: list[int]
    kernel_size: list[int]
    data_t: FixedPoint = FixedPoint(16, 8)

    # class variables
    name: ClassVar[str] = "sliding_window"
    register: ClassVar[bool] = False

    def __post_init__(self):
        assert len(self.kernel_size) == min(self.dimensionality).value, \
                f"{self.__class__} only supports {min(self.dimensionality).value}D kernel size"
        assert len(self.stride) == min(self.dimensionality).value, \
                f"{self.__class__} only supports {min(self.dimensionality).value}D stride"
        assert len(self.pad) == 2*min(self.dimensionality).value, \
                f"{self.__class__} only supports {min(self.dimensionality).value}D pad"
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
            simd_lanes=[*self.kernel_size],
            data_type=self.data_t,
            buffer_depth=2,
            name="io_out"
        )]

    @property
    def rate_in(self) -> list[float]:
        # get the input size
        input_size = math.prod(self.input_iter_space[0][:-1])
        # get the padded input size
        padded_input_size = math.prod([ self.input_iter_space[0][i] + \
                self.pad[2*i] + self.pad[2*i+1] for i in range(len(self.input_iter_space[0][:-1])) ])
        # ratio between shapes
        return [ input_size/float(padded_input_size) ]

    @property
    def rate_out(self) -> list[float]:
        return [ math.prod(self.output_iter_space[0])/float(math.prod(self.input_iter_space[0])) ]

@dataclass
class SlidingWindowHLS(SlidingWindowHLSBase): #FIXME

    register: ClassVar[bool] = True

    @property
    def rows_out(self) -> int:
        return math.ceil((self.rows-self.kernel_size[0]+1)/self.stride[0])

    @property
    def cols_out(self) -> int:
        return math.ceil((self.cols-self.kernel_size[1]+1)/self.stride[1])

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.channels] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows_out, self.cols_out, self.channels] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.channels, *self.pad,
                *self.stride, *self.kernel_size, self.data_t.width ]


@dataclass
class SlidingWindowHLS3D(ModuleHLS3DBase, SlidingWindowHLSBase): #FIXME

    register: ClassVar[bool] = True

    @property
    def rows_out(self) -> int:
        return math.ceil((self.rows-self.kernel_size[0]+1)/self.stride[0])

    @property
    def cols_out(self) -> int:
        return math.ceil((self.cols-self.kernel_size[1]+1)/self.stride[1])

    @property
    def depth_out(self) -> int:
        return math.ceil((self.depth-self.kernel_size[2]+1)/self.stride[2])

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.depth, self.channels] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows_out, self.cols_out, self.depth_out, self.channels] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.depth, self.channels, *self.pad,
                *self.stride, *self.kernel_size, self.data_t.width ]

    #def functional_model(self, data):
    #    # check input dimensionality
    #    batch_size = data.shape[0]
    #    assert data.shape[1] == self.rows    , "ERROR: invalid row dimension"
    #    assert data.shape[2] == self.cols    , "ERROR: invalid column dimension"
    #    assert data.shape[3] == self.depth    , "ERROR: invalid depth dimension"
    #    assert data.shape[4] == self.channels, "ERROR: invalid channel dimension"

    #    #pad input
    #    data_padded = np.ndarray((
    #        batch_size,
    #        self.rows + self.pad_bottom + self.pad_top,
    #        self.cols + self.pad_left + self.pad_right,
    #        self.depth + self.pad_back + self.pad_front,
    #        self.channels),dtype=float)

    #    for index,_ in np.ndenumerate(data_padded):
    #        if  (index[1] < self.pad_bottom):
    #            data_padded[index] = 0
    #        elif(index[2] < self.pad_left):
    #            data_padded[index] = 0
    #        elif(index[3] < self.pad_back):
    #            data_padded[index] = 0
    #        elif(index[1] > self.rows - 1 + self.pad_bottom):
    #            data_padded[index] = 0
    #        elif(index[2] > self.cols - 1 + self.pad_left):
    #            data_padded[index] = 0
    #        elif(index[3] > self.depth - 1 + self.pad_back):
    #            data_padded[index] = 0
    #        else:
    #            data_padded[index] = data[
    #                index[0],
    #                index[1] - self.pad_left,
    #                index[2] - self.pad_bottom,
    #                index[3] - self.pad_back,
    #                index[4]]

    #    out = np.ndarray((
    #        batch_size,
    #        self.rows_out(),
    #        self.cols_out(),
    #        self.depth_out(),
    #        self.channels,
    #        self.kernel_rows,
    #        self.kernel_cols,
    #        self.kernel_depth),dtype=float)

    #    for index,_ in np.ndenumerate(out):
    #        out[index] = data_padded[
    #            index[0],
    #            index[1]*self.stride_rows+index[5],
    #            index[2]*self.stride_cols+index[6],
    #            index[3]*self.stride_depth+index[7],
    #            index[4]]

    #    return out

try:
    DEFAULT_SLIDING_WINDOW_RSC_MODELS: dict[str, ResourceModel] = { rsc_type: get_cached_resource_model(SlidingWindowHLS,
                                    rsc_type, "default") for rsc_type in DEFAULT_HLS_PLATFORM.resource_types }
except FileNotFoundError:
    print("CRITICAL WARNING: default resource models not found for SlidingWindow, default resource modelling will fail")

@eval_resource_model.register
def _(m: SlidingWindowHLS, rsc_type: str, _model: Optional[ResourceModel] = None) -> int:

    # get the resource model
    model: ResourceModel = _model if _model is not None else DEFAULT_SLIDING_WINDOW_RSC_MODELS[rsc_type]

    # check the correct resource type
    assert rsc_type == model.rsc_type, f"Incompatible resource type with model: {rsc_type}"

    # get the resource model
    match rsc_type:
        case "DSP":
            return 0
        case _:
            return model(m)

