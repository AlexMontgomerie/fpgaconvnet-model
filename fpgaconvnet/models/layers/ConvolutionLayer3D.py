import importlib
import math
from typing import Union, List
from dataclasses import dataclass, field

import pydot
import numpy as np

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model
from fpgaconvnet.models.layers import Layer3D

from fpgaconvnet.models.modules import SlidingWindow3D
from fpgaconvnet.models.modules import VectorDot3D
from fpgaconvnet.models.modules import Conv3D
from fpgaconvnet.models.modules import Squeeze3D
from fpgaconvnet.models.modules import Fork3D
from fpgaconvnet.models.modules import Accum3D
from fpgaconvnet.models.modules import Glue3D
from fpgaconvnet.models.modules import Bias3D
from fpgaconvnet.models.modules import Pad3D
from fpgaconvnet.models.modules import ShiftScale3D

@dataclass(kw_only=True)
class ConvolutionLayer3D(Layer3D):
    filters: int
    kernel_rows: int
    kernel_cols: int
    kernel_depth: int
    coarse_in: int = 1
    coarse_out: int = 1
    coarse_group: int = 1
    stride_rows: int = 2
    stride_cols: int = 2
    stride_depth: int = 2
    groups: int = 1
    pad_top: int = 0
    pad_right: int = 0
    pad_front: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_back: int = 0
    fine: int  = 1
    input_t: FixedPoint = field(default_factory=lambda: FixedPoint(16,8), init=True)
    output_t: FixedPoint = field(default_factory=lambda: FixedPoint(16,8), init=True)
    weight_t: FixedPoint = field(default_factory=lambda: FixedPoint(16,8), init=True)
    acc_t: FixedPoint = field(default_factory=lambda: FixedPoint(32,16), init=True)
    has_bias: int = 0 # default to no bias for old configs
    sparsity: list = field(default_factory=lambda: [], init=True)
    block_floating_point: bool = False
    backend: str = "chisel" # default to no bias for old configs
    regression_model: str = "linear_regression"
    stream_weights: int = 0
    stream_inputs: list = field(default_factory=lambda: [0], init=True)

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        # check if the layer is depthwise
        self.depthwise = (self.groups == self.channels) and (self.groups == self.filters)
        self.pointwise = np.prod(self.kernel_size) == 1

        # save sparsity
        if len(self.sparsity) > 0:
            # reject if pointwise or low sparsity
            if self.pointwise or np.mean(self.sparsity) < 0.1:
                self.sparsity = []
        # self.sparsity = sparsity

        # weights buffering flag
        if self.backend == "hls":
            self.double_buffered = False
            self.stream_weights = 0
            self.data_packing = False
            self.use_uram = False
        elif self.backend == "chisel":
            self.double_buffered = False
            self.data_packing = True
            self.use_uram = False

        # regression model
        assert self.regression_model in ["linear_regression", "xgboost"], f"{self.regression_model} is an invalid regression model"
        self.regression_model = self.regression_model

        self.modules["pad3d"] = Pad3D(
                self.rows_in(), self.cols_in(), self.depth_in(),
                self.channels_in()//(self.coarse_in*self.coarse_group),
                self.pad_top, self.pad_bottom, self.pad_left, self.pad_right,
                self.pad_front, self.pad_back, backend=self.backend,
                regression_model=self.regression_model)

        self.modules["sliding_window3d"] = SlidingWindow3D(
                self.rows_in() +self.pad_top + self.pad_bottom,
                self.cols_in() + self.pad_left + self.pad_right,
                self.depth_in() + self.pad_front + self.pad_back,
                self.channels_in()//(self.coarse_in*self.coarse_group),
                self.kernel_rows, self.kernel_cols, self.kernel_depth,
                self.stride_rows, self.stride_cols, self.stride_depth,
                0, 0, 0, 0, 0, 0, backend=self.backend,
                regression_model=self.regression_model)

        if self.backend == "hls":

            self.modules["fork3d"] = Fork3D(
                    self.rows_out(), self.cols_out(), self.depth_out(),
                    self.channels_in()//(self.coarse_in*self.coarse_group),
                    self.kernel_rows, self.kernel_cols, self.kernel_depth,
                    self.coarse_out, backend=self.backend, regression_model=self.regression_model)

            self.modules["conv3d"] = Conv3D(
                    self.rows_out(), self.cols_out(), self.depth_out(),
                    self.channels_in()//(self.coarse_in*self.coarse_group),
                    self.filters//(self.coarse_out*self.groups),
                    self.kernel_rows, self.kernel_cols, self.kernel_depth,
                    backend=self.backend, regression_model=self.regression_model)

            self.modules["accum3d"] = Accum3D(
                    self.rows_out(), self.cols_out(), self.depth_out(),
                    self.channels_in()//(self.coarse_in*self.groups),
                    self.filters//(self.coarse_out*self.groups), 1,
                    backend=self.backend, regression_model=self.regression_model)

        elif self.backend == "chisel":

            self.modules["squeeze3d"] = Squeeze3D(
                    self.rows_out(), self.cols_out(), self.depth_out(),
                    self.channels_in()//(self.coarse_in*self.coarse_group),
                    self.kernel_rows*self.kernel_cols*self.kernel_depth,
                    self.fine, backend=self.backend, regression_model=self.regression_model)

            self.modules["fork3d"] = Fork3D(
                    self.rows_out(), self.cols_out(), self.depth_out(),
                    self.channels_in()//(self.coarse_in*self.coarse_group),
                    self.fine, 1, 1, self.coarse_out, backend=self.backend,
                    regression_model=self.regression_model)

            if len(self.sparsity) == 0:
                self.modules["vector_dot3d"] = VectorDot3D(
                        self.rows_out(), self.cols_out(), self.depth_out(),
                        self.channels_in()//(self.coarse_in*self.coarse_group),
                        self.filters//(self.coarse_out*self.groups), self.fine,
                        backend=self.backend, regression_model=self.regression_model)

                self.modules["accum3d"] = Accum3D(
                        self.rows_out(), self.cols_out(), self.depth_out(),
                        (self.kernel_rows*self.kernel_cols*self.kernel_depth*self.channels_in()
                            )//(self.fine*self.coarse_in*self.groups),
                        self.filters//(self.coarse_out*self.groups), 1,
                        backend=self.backend, regression_model=self.regression_model)
            else:
                raise NotImplementedError

        self.modules["glue3d"] = Glue3D(
                self.rows_out(), self.cols_out(), self.depth_out(),
                1, int(self.filters/self.coarse_out), self.coarse_in,
                self.coarse_out, backend=self.backend,
                regression_model=self.regression_model) # TODO

        self.modules["bias3d"] = Bias3D(
                self.rows_out(), self.cols_out(), self.depth_out(),
                1, self.filters, backend=self.backend,
                regression_model=self.regression_model) # TODO

        self.modules["shift_scale3d"] = ShiftScale3D(self.rows_out(), self.cols_out(), self.depth_out(), 1, self.filters//(self.coarse_out*self.coarse_group))

        # update modules
        self.update()

    @property
    def kernel_size(self) -> List[int]:
        return [ self.kernel_rows, self.kernel_cols, self.kernel_depth]

    @property
    def stride(self) -> List[int]:
        return [ self.stride_rows, self.stride_cols, self.stride_depth]

    @property
    def pad(self) -> List[int]:
        return [
            self.pad_top,
            self.pad_left,
            self.pad_front,
            self.pad_bottom,
            self.pad_right,
            self.pad_back,
        ]

    @kernel_size.setter
    def kernel_size(self, val: List[int]) -> None:
        assert(len(val) == 3, "kernel size must be a list of three integers")
        self.kernel_rows = val[0]
        self.kernel_cols = val[1]
        self.kernel_depth = val[2]

    @stride.setter
    def stride(self, val: List[int]) -> None:
        assert(len(val) == 3, "stride must be a list of three integers")
        self.stride_rows = val[0]
        self.stride_cols = val[1]
        self.stride_depth = val[2]

    @pad.setter
    def pad(self, val: List[int]) -> None:
        assert(len(val) == 6, "pad must be a list of six integers")
        self.pad_top    = val[0]
        self.pad_right  = val[4]
        self.pad_bottom = val[3]
        self.pad_left   = val[1]
        self.pad_front  = val[2]
        self.pad_back   = val[5]

    def rows_out(self) -> int:
        return self.modules["sliding_window3d"].rows_out()

    def cols_out(self) -> int:
        return self.modules["sliding_window3d"].cols_out()

    def depth_out(self) -> int:
        return self.modules["sliding_window3d"].depth_out()

    def channels_out(self) -> int:
        return self.filters

    def streams_in(self) -> int:
        """
        Returns
        -------
        int
            number of parallel streams into the layer.
        """
        return self.coarse_in*self.coarse_group

    def streams_out(self) -> int:
        """
        Returns
        -------
        int
            number of parallel streams out of the layer.
        """
        return self.coarse_out*self.coarse_group

    def get_stream_sparsity(self, interleave=True):
        raise NotImplementedError

    def pipeline_depth(self):
        # pipeline depth of the sliding window
        # minus the total words in the pipeline from padding (missing)
        # plus the words needed to fill the accum buffer

        # - ( self.pad_top * self.cols * self.channels//self.streams_in() + \
        #         (self.pad_left+self.pad_right)*self.channels//self.streams_in() )
        return (self.kernel_rows-1)*(self.cols+self.pad_left+self.pad_right)*(self.depth+self.pad_front+self.pad_back)*self.channels//self.streams_in() + \
                (self.kernel_cols-1)*(self.depth+self.pad_front+self.pad_back)*self.channels//self.streams_in() + \
                (self.kernel_depth-1)*self.channels//self.streams_in() + \
                self.channels//self.streams_in()

    def update(self):

        # pad
        self.modules['pad3d'].rows     = self.rows
        self.modules['pad3d'].cols     = self.cols
        self.modules['pad3d'].depth    = self.depth
        self.modules['pad3d'].channels = self.channels//(self.coarse_in*self.coarse_group)
        self.modules['pad3d'].data_width = self.input_t.width
        self.modules['pad3d'].streams = self.coarse_in*self.coarse_group
        self.modules['pad3d'].pad_top = self.pad_top
        self.modules['pad3d'].pad_bottom = self.pad_bottom
        self.modules['pad3d'].pad_left = self.pad_left
        self.modules['pad3d'].pad_right = self.pad_right
        self.modules['pad3d'].pad_front = self.pad_front
        self.modules['pad3d'].pad_back = self.pad_back

        # sliding window
        self.modules['sliding_window3d'].rows     = self.rows + self.pad_top + self.pad_bottom
        self.modules['sliding_window3d'].cols     = self.cols + self.pad_left + self.pad_right
        self.modules['sliding_window3d'].depth    = self.depth + self.pad_front + self.pad_back
        self.modules['sliding_window3d'].channels = self.channels//self.streams_in()
        self.modules['sliding_window3d'].kernel_cols = self.kernel_cols
        self.modules['sliding_window3d'].kernel_rows = self.kernel_rows
        self.modules['sliding_window3d'].kernel_depth= self.kernel_depth
        self.modules['sliding_window3d'].stride_cols = self.stride_cols
        self.modules['sliding_window3d'].stride_rows = self.stride_rows
        self.modules['sliding_window3d'].stride_depth= self.stride_depth
        self.modules['sliding_window3d'].data_width = self.input_t.width
        if self.data_packing:
            self.modules['sliding_window3d'].streams = self.coarse_in*self.coarse_group
        self.modules['sliding_window3d'].pad_top = 0
        self.modules['sliding_window3d'].pad_bottom = 0
        self.modules['sliding_window3d'].pad_left = 0
        self.modules['sliding_window3d'].pad_right = 0
        self.modules['sliding_window3d'].pad_front = 0
        self.modules['sliding_window3d'].pad_back = 0

        if self.backend == "chisel":
            # squeeze3d
            self.modules['squeeze3d'].rows     = self.rows_out()
            self.modules['squeeze3d'].cols     = self.cols_out()
            self.modules['squeeze3d'].depth    = self.depth_out()
            self.modules['squeeze3d'].channels = self.channels//(self.coarse_in*self.coarse_group)
            self.modules['squeeze3d'].coarse_in  = self.kernel_rows*self.kernel_cols*self.kernel_depth
            self.modules['squeeze3d'].coarse_out = self.fine
            self.modules['squeeze3d'].data_width = self.input_t.width
            if self.data_packing:
                self.modules['squeeze3d'].streams = self.coarse_in*self.coarse_group

        # fork3d
        self.modules['fork3d'].rows     = self.rows_out()
        self.modules['fork3d'].cols     = self.cols_out()
        self.modules['fork3d'].depth    = self.depth_out()
        self.modules['fork3d'].channels = self.channels_in()//(self.coarse_in*self.coarse_group)
        self.modules['fork3d'].coarse   = self.coarse_out
        self.modules['fork3d'].data_width     = self.input_t.width
        if self.data_packing:
            self.modules['fork3d'].streams = self.coarse_in*self.coarse_group
        if self.backend == "chisel":
            self.modules['fork3d'].kernel_rows = self.fine
            self.modules['fork3d'].kernel_cols = 1
            self.modules['fork3d'].kernel_depth = 1

        if self.backend == "hls":
            # TODO: check the group parameter
            self.modules['conv3d'].rows     = self.rows_out()
            self.modules['conv3d'].cols     = self.cols_out()
            self.modules['conv3d'].depth    = self.depth_out()
            self.modules['conv3d'].channels = self.channels_in()//(self.coarse_in*self.coarse_group)
            self.modules['conv3d'].filters  = self.filters//(self.coarse_out*self.coarse_group)
            self.modules['conv3d'].fine     = self.fine
            self.modules['conv3d'].data_width     = self.input_t.width
            self.modules['conv3d'].weight_width   = self.weight_t.width
            self.modules['conv3d'].acc_width      = self.acc_t.width
        elif self.backend == "chisel":
            # kernel dot
            self.modules['vector_dot3d'].rows     = self.rows_out()
            self.modules['vector_dot3d'].cols     = self.cols_out()
            self.modules['vector_dot3d'].depth    = self.depth_out()
            self.modules['vector_dot3d'].channels = (
                    self.channels*self.kernel_rows*self.kernel_cols*self.kernel_depth)//(
                    self.fine*self.coarse_in*self.groups)
            self.modules['vector_dot3d'].filters  = self.filters//(self.coarse_out*self.coarse_group)
            self.modules['vector_dot3d'].fine     = self.fine
            self.modules['vector_dot3d'].data_width     = self.input_t.width
            self.modules['vector_dot3d'].weight_width   = self.weight_t.width
            self.modules['vector_dot3d'].acc_width      = self.acc_t.width
            if self.data_packing:
                self.modules['vector_dot3d'].streams = self.coarse_in*self.coarse_group*self.coarse_out

        # accum3d
        self.modules['accum3d'].rows     = self.rows_out()
        self.modules['accum3d'].cols     = self.cols_out()
        self.modules['accum3d'].depth    = self.depth_out()
        self.modules['accum3d'].filters  = self.filters//(self.coarse_out*self.coarse_group)
        self.modules['accum3d'].data_width    = self.acc_t.width
        if self.data_packing:
            self.modules['accum3d'].streams = self.coarse_in*self.coarse_group*self.coarse_out
        if self.backend == "hls":
            # TODO: check the group parameter
            self.modules['accum3d'].channels  = self.channels_in()//(self.coarse_in*self.coarse_group)
        elif self.backend == "chisel":
            self.modules['accum3d'].channels = (
                    self.channels*self.kernel_rows*self.kernel_cols*self.kernel_depth)//(
                    self.fine*self.coarse_in*self.coarse_group)
            self.modules['accum3d'].groups   = 1

        # glue3d
        self.modules['glue3d'].rows       = self.rows_out()
        self.modules['glue3d'].cols       = self.cols_out()
        self.modules['glue3d'].depth      = self.depth_out()
        self.modules['glue3d'].filters    = self.filters//self.coarse_group
        self.modules['glue3d'].coarse_in  = self.coarse_in
        self.modules['glue3d'].coarse_out = self.coarse_out
        self.modules['glue3d'].data_width = self.acc_t.width
        self.modules['glue3d'].streams = self.coarse_group*self.coarse_out

        # bias3d
        self.modules['bias3d'].rows           = self.rows_out()
        self.modules['bias3d'].cols           = self.cols_out()
        self.modules['bias3d'].depth          = self.depth_out()
        self.modules['bias3d'].filters        = self.filters//(self.coarse_group*self.coarse_out)
        self.modules['bias3d'].data_width     = self.output_t.width
        self.modules['bias3d'].biases_width   = self.acc_t.width
        if self.data_packing:
            self.modules['bias3d'].streams = self.coarse_out*self.coarse_group

        # shift scale
        self.modules['shift_scale3d'].rows           = self.rows_out()
        self.modules['shift_scale3d'].cols           = self.cols_out()
        self.modules['shift_scale3d'].filters        = self.filters//(self.coarse_out*self.coarse_group)
        self.modules['shift_scale3d'].data_width     = self.output_t.width
        self.modules['shift_scale3d'].biases_width   = self.acc_t.width
        if self.data_packing:
            self.modules['shift_scale3d'].streams = self.coarse_out*self.coarse_group

    def layer_info(self,parameters,batch_size=1):
        Layer3D.layer_info(self, parameters, batch_size)
        parameters.filters      = self.filters
        parameters.groups       = self.groups
        parameters.coarse_group = self.coarse_group
        parameters.fine         = self.fine
        parameters.sparsity     = 0
        parameters.kernel_size.extend(self.kernel_size)
        parameters.kernel_rows = self.kernel_rows
        parameters.kernel_cols = self.kernel_cols
        parameters.kernel_depth = self.kernel_depth
        parameters.stride.extend(self.stride)
        parameters.stride_rows  = self.stride_rows
        parameters.stride_cols  = self.stride_cols
        parameters.stride_depth = self.stride_depth
        parameters.pad_top      = self.pad_top
        parameters.pad_right    = self.pad_right
        parameters.pad_front    = self.pad_front
        parameters.pad_bottom   = self.pad_bottom
        parameters.pad_left     = self.pad_left
        parameters.pad_back     = self.pad_back
        parameters.has_bias     = self.has_bias
        parameters.block_floating_point    = self.block_floating_point
        self.input_t.to_protobuf(parameters.input_t)
        self.output_t.to_protobuf(parameters.output_t)
        self.weight_t.to_protobuf(parameters.weight_t)
        self.acc_t.to_protobuf(parameters.acc_t)
        parameters.data_t.Clear()
        parameters.use_uram     = self.use_uram

    def get_coarse_group_feasible(self):
        return get_factors(self.groups)

    def get_coarse_in_feasible(self):
        return get_factors(int(self.channels_in())//self.groups)

    def get_coarse_out_feasible(self):
        return get_factors(int(self.channels_out())//self.groups)

    def get_fine_feasible(self):
        if self.backend == "chisel":
            return get_factors(self.kernel_rows*self.kernel_cols*self.kernel_depth)
        elif self.backend == "hls":
            if self.kernel_depth != self.kernel_rows and self.kernel_rows == self.kernel_cols:
                if self.kernel_depth == 1:
                    fine_feasible = [1, self.kernel_rows, self.kernel_rows * self.kernel_cols]
                elif self.kernel_rows == 1:
                    fine_feasible = [1, self.kernel_depth]
                else:
                    fine_feasible = [
                        1,
                        self.kernel_depth,
                        self.kernel_rows,
                        self.kernel_depth * self.kernel_rows,
                        self.kernel_rows * self.kernel_cols,
                        self.kernel_depth * self.kernel_rows * self.kernel_cols,
                    ]
            elif self.kernel_depth == self.kernel_rows and self.kernel_rows == self.kernel_cols:
                if self.kernel_depth == 1:
                    fine_feasible = [1]
                else:
                    fine_feasible = [
                        1,
                        self.kernel_depth,
                        self.kernel_depth * self.kernel_rows,
                        self.kernel_depth * self.kernel_rows * self.kernel_cols,
                    ]
            else:
                fine_feasible = [
                    1,
                    self.kernel_depth,
                    self.kernel_rows,
                    self.kernel_cols,
                    self.kernel_depth * self.kernel_rows,
                    self.kernel_depth * self.kernel_cols,
                    self.kernel_rows * self.kernel_cols,
                    self.kernel_depth * self.kernel_rows * self.kernel_cols,
                ]
            return fine_feasible

    def get_weights_reloading_feasible(self):
        return get_factors(self.filters//(self.groups*self.coarse_out))

    def get_parameters_size(self):
        weights_size = self.channels_in() * ( self.filters // self.groups ) * self.kernel_rows * self.kernel_cols * self.kernel_depth
        bias_size = 0
        return {
            "weights"   : weights_size,
            "bias"      : bias_size
        }

    def get_operations(self):
        ops = self.kernel_rows*self.kernel_cols*self.kernel_depth*self.channels_in()*self.filters*self.rows_out()*self.cols_out()*self.depth_out()
        if self.has_bias:
            ops += self.filters*self.rows_out()*self.cols_out()*self.depth_out()
        return ops

    def resource(self):

        if self.backend == "chisel":

            # get module resource models
            sw_rsc          = self.modules['sliding_window3d'].rsc()
            squeeze_rsc     = self.modules['squeeze3d'].rsc()
            fork_rsc        = self.modules['fork3d'].rsc()
            vector_dot_rsc  = self.modules['vector_dot3d'].rsc()
            accum_rsc       = self.modules['accum3d'].rsc()
            glue_rsc        = self.modules['glue3d'].rsc()
            bias_rsc        = self.modules['bias3d'].rsc()
            shift_scale_rsc = self.modules['shift_scale3d'].rsc()

            # remove redundant modules
            if self.kernel_rows == 1 and self.kernel_cols == 1 and self.kernel_depth == 1:
                sw_rsc      = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            if self.fine == self.kernel_rows*self.kernel_cols*self.kernel_depth:
                squeeze_rsc = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            if self.coarse_out == 1:
                fork_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            if int(self.channels_in()/(self.coarse_in*self.coarse_group)) == 1:
                accum_rsc   = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            if self.coarse_in == 1:
                glue_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            if self.has_bias == 0:
                bias_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            if not self.block_floating_point:
                shift_scale_rsc = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}

            # accumulate resource usage based on coarse factors
            # rsc = { rsc_type: (
            #     sw_rsc[rsc_type]*self.coarse_in*self.coarse_group +
            #     squeeze_rsc[rsc_type]*self.coarse_in*self.coarse_group +
            #     fork_rsc[rsc_type]*self.coarse_in*self.coarse_group +
            #     vector_dot_rsc[rsc_type]*self.coarse_in*self.coarse_out*self.coarse_group +
            #     accum_rsc[rsc_type]*self.coarse_in*self.coarse_out*self.coarse_group +
            #     glue_rsc[rsc_type]*self.coarse_out*self.coarse_group +
            #     bias_rsc[rsc_type]*self.coarse_out
            # ) for rsc_type in ["LUT", "FF", "DSP", "BRAM"] }

            # dsp packing
            if self.weight_t.width <= 4 and self.input_t.width <= 4:
                vector_dot_rsc["DSP"] = vector_dot_rsc["DSP"]*0.25
            elif self.weight_t.width <= 8 and self.input_t.width <= 8:
                vector_dot_rsc["DSP"] = vector_dot_rsc["DSP"]*0.5

            if self.data_packing:
                rsc = { rsc_type: (
                    sw_rsc[rsc_type] +
                    squeeze_rsc[rsc_type] +
                    fork_rsc[rsc_type] +
                    math.ceil(vector_dot_rsc[rsc_type]) +
                    accum_rsc[rsc_type] +
                    glue_rsc[rsc_type] +
                    bias_rsc[rsc_type] +
                    shift_scale_rsc[rsc_type]
                ) for rsc_type in ["LUT", "FF", "DSP", "BRAM"] }
            else:
                rsc = { rsc_type: (
                    sw_rsc[rsc_type] +
                    squeeze_rsc[rsc_type] +
                    fork_rsc[rsc_type] +
                    vector_dot_rsc[rsc_type] +
                    accum_rsc[rsc_type] +
                    glue_rsc[rsc_type] +
                    bias_rsc[rsc_type]*self.coarse_out*self.coarse_group +
                    shift_scale_rsc[rsc_type]*self.coarse_out*self.coarse_group
                ) for rsc_type in ["LUT", "FF", "DSP", "BRAM"] }

        elif self.backend == "hls":

            # get module resource models
            sw_rsc          = self.modules['sliding_window3d'].rsc()
            fork_rsc        = self.modules['fork3d'].rsc()
            conv_rsc        = self.modules['conv3d'].rsc()
            accum_rsc       = self.modules['accum3d'].rsc()
            glue_rsc        = self.modules['glue3d'].rsc()
            bias_rsc        = self.modules['bias3d'].rsc()

            # remove redundant modules
            if self.kernel_rows == 1 and self.kernel_cols == 1 and self.kernel_depth == 1:
                sw_rsc      = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            if self.coarse_out == 1:
                fork_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            if int(self.channels_in()/(self.coarse_in*self.coarse_group)) == 1:
                accum_rsc   = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            if self.coarse_in == 1:
                glue_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            if self.has_bias:
                bias_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}

            # accumulate resource usage based on coarse factors
            rsc = { rsc_type: (
                sw_rsc[rsc_type]*self.coarse_in*self.coarse_group +
                fork_rsc[rsc_type]*self.coarse_in*self.coarse_group +
                conv_rsc[rsc_type]*self.coarse_in*self.coarse_out*self.coarse_group +
                accum_rsc[rsc_type]*self.coarse_in*self.coarse_out*self.coarse_group +
                glue_rsc[rsc_type]*self.coarse_out*self.coarse_group +
                bias_rsc[rsc_type]*self.coarse_out
            ) for rsc_type in ["LUT", "FF", "DSP", "BRAM"] }

        # weight usage
        weight_memory_depth = float((self.filters/self.groups)* \
                                    self.channels_in()* \
                                    self.kernel_rows* \
                                    self.kernel_cols* \
                                    self.kernel_depth) / \
            float(self.fine*self.coarse_in*self.coarse_out*self.coarse_group)

        # apply double buffering
        if self.double_buffered:
            weight_memory_depth *= 2

        if self.data_packing:
            weight_array_depth = math.ceil(weight_memory_depth)
            if len(self.sparsity) == 0:
                weight_array_width = self.weight_t.width*self.fine*self.coarse_in*self.coarse_out*self.coarse_group
                weight_array_num = 1
            else:
                weight_array_width = self.weight_t.width*self.fine*self.coarse_out*self.coarse_group
                weight_array_num = self.coarse_in
        else:
            weight_array_depth = math.ceil(weight_memory_depth)
            weight_array_width = self.weight_t.width
            weight_array_num = self.fine*self.coarse_in*self.coarse_out*self.coarse_group

        weights_bram_usage, weights_uram_usage = self.stream_rsc(weight_array_depth, weight_array_width, weight_array_num)

        # if streaming weights, set to zero
        if self.stream_weights:
            weights_bram_usage = 0

        # bias usage
        if self.has_bias:
            bias_memory_depth =  math.ceil(float(self.filters) / float(self.coarse_out*self.coarse_group))
            if self.data_packing:
                bias_array_width = self.acc_t.width*self.coarse_out*self.coarse_group
                bias_array_num = 1
            else:
                bias_array_width = self.acc_t.width
                bias_array_num = self.coarse_out*self.coarse_group
            biases_bram_usage = bram_array_resource_model(
                        bias_memory_depth, bias_array_width,
                        "memory") * bias_array_num
        else:
            biases_bram_usage = 0

        # bfp shift scale usage
        if self.block_floating_point:
            shift_scale_memory_depth = float(self.filters) / float(self.coarse_out*self.coarse_group)
            shift_scale_bram_usage = bram_array_resource_model(
                        int(shift_scale_memory_depth),self.acc_t.width,
                        "memory") * self.coarse_out * self.coarse_group * 2
        else:
            shift_scale_bram_usage = 0

        # add weights and bias to resources
        rsc["BRAM"] += weights_bram_usage + biases_bram_usage + shift_scale_bram_usage
        rsc["URAM"] = weights_uram_usage

        # return total resource
        return rsc

    from fpgaconvnet.models.layers.utils import stream_unit, stream_step
    from fpgaconvnet.models.layers.utils import off_chip_addr_range, on_chip_addr_range, off_chip_buffer_size
    from fpgaconvnet.models.layers.utils import stream_bits, stream_cycles, stream_bw
    from fpgaconvnet.models.layers.utils import stream_rsc, stream_buffer

    def visualise(self, name):
        cluster = pydot.Cluster(name, label=name,
                style="dashed", bgcolor="lightpink")

        # names
        slwin_name = [[""]*self.coarse_in]*self.coarse_group
        fork_name = [[""]*self.coarse_in]*self.coarse_group
        # conv_name = [[[""]*self.coarse_in]*self.coarse_out]*self.coarse_group
        vector_dot_name = [[[""]*self.coarse_in]*self.coarse_out]*self.coarse_group
        accum_name = [[[""]*self.coarse_in]*self.coarse_out]*self.coarse_group
        glue_name = [[""]*self.coarse_out]*self.coarse_group
        bias_name = [[""]*self.coarse_out]*self.coarse_group

        for g in range(self.coarse_group):
            for i in range(self.coarse_in):
                # define names
                slwin_name[g][i] = "_".join([name, "sw3d", str(g), str(i)])
                fork_name[g][i] = "_".join([name, "fork3d", str(g), str(i)])
                # add nodes
                cluster.add_node(self.modules["sliding_window3d"].visualise(slwin_name[g][i]))
                cluster.add_node(self.modules["fork3d"].visualise(fork_name[g][i]))
                # add edges
                cluster.add_edge(pydot.Edge(slwin_name[g][i], fork_name[g][i]))

                # iterate over coarse out
                for j in range(self.coarse_out):
                    # define names
                    # conv_name[g][j][i] = "_".join([name, "conv3d", str(g), str(j), str(i)])
                    vector_dot_name[g][j][i] = "_".join([name, "vector_dot3d", str(g), str(j), str(i)])
                    accum_name[g][j][i] = "_".join([name, "accum3d", str(g), str(j), str(i)])
                    glue_name[g][j] = "_".join([name, "glue3d", str(g), str(j)])
                    bias_name[g][j] = "_".join([name, "bias3d", str(g), str(j)])

                    # add nodes
                    # cluster.add_node(self.modules["conv3d"].visualise(conv_name[g][j][i]))
                    cluster.add_node(self.modules["vector_dot3d"].visualise(vector_dot_name[g][j][i]))
                    cluster.add_node(self.modules["accum3d"].visualise(accum_name[g][j][i]))

                    # add edges
                    # cluster.add_edge(pydot.Edge(fork_name[g][i], conv_name[g][j][i]))
                    cluster.add_edge(pydot.Edge(fork_name[g][i], vector_dot_name[g][j][i]))
                    # cluster.add_edge(pydot.Edge(conv_name[g][j][i], accum_name[g][j][i]))
                    cluster.add_edge(pydot.Edge(vector_dot_name[g][j][i], accum_name[g][j][i]))
                    cluster.add_edge(pydot.Edge(accum_name[g][j][i], glue_name[g][j]))

        for g in range(self.coarse_group):
            for j in range(self.coarse_out):

                # add nodes
                cluster.add_node(self.modules["glue3d"].visualise(glue_name[g][j]))
                cluster.add_node(self.modules["bias3d"].visualise(bias_name[g][j]))

                # add edges
                cluster.add_edge(pydot.Edge(glue_name[g][j], bias_name[g][j]))


        return cluster, np.array(slwin_name).flatten().tolist(), np.array(bias_name).flatten().tolist()

    def functional_model(self,data,weights,bias,batch_size=1):
        import torch

        assert data.shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.depth_in()   , "ERROR (data): invalid depth dimension"
        assert data.shape[3] == self.channels_in(), "ERROR (data): invalid channel dimension"

        assert weights.shape[0] == self.filters ,   "ERROR (weights): invalid filter dimension"
        assert weights.shape[1] == self.channels//self.groups,\
                                                    "ERROR (weights): invalid channel dimension"
        assert weights.shape[2] == self.kernel_rows,\
                                                    "ERROR (weights): invalid kernel dimension"
        assert weights.shape[3] == self.kernel_cols,\
                                                    "ERROR (weights): invalid kernel dimension"
        assert weights.shape[4] == self.kernel_depth,\
                                                    "ERROR (weights): invalid kernel dimension"

        assert bias.shape[0] == self.filters  ,     "ERROR (bias): invalid filter dimension"

        # instantiate convolution layer
        # convolution_layer = torch.nn.Conv3d(self.channels_in(), self.filters, (self.kernel_depth, self.kernel_rows, self.kernel_cols), stride=(self.stride_depth, self.stride_rows, self.stride_cols), padding=(self.pad_front, self.pad_top, self.pad_right), groups=self.groups, bias=True)
        convolution_layer = torch.nn.Conv3d(self.channels_in(), self.filters, (self.kernel_depth, self.kernel_rows, self.kernel_cols), stride=(self.stride_depth, self.stride_rows, self.stride_cols), padding=0, groups=self.groups, bias=True)

        # update weights
        convolution_layer.weight = torch.nn.Parameter(
                torch.from_numpy(np.moveaxis(weights,-1,-3)))

        # update bias
        convolution_layer.bias = torch.nn.Parameter(torch.from_numpy(bias))

        # get the padding
        padding = [
            self.pad_left,
            self.pad_right,
            self.pad_top,
            self.pad_bottom,
            self.pad_front,
            self.pad_back,
        ]

        # return output featuremap
        data = np.moveaxis(data, [-1, -2], [0, 1])
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        data = torch.nn.functional.pad(torch.from_numpy(data), padding, "constant", 0.0)
        data = convolution_layer(data).detach().numpy()
        print(data.shape)
        return data
        # return convolution_layer(data).detach().numpy()

