import importlib
import math
from typing import Union, List

import pydot
import numpy as np

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model
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

class ConvolutionLayer3D(Layer3D):

    def __init__(
            self,
            filters: int,
            rows: int,
            cols: int,
            depth: int,
            channels: int,
            coarse_in: int = 1,
            coarse_out: int = 1,
            coarse_group: int = 1,
            kernel_rows: int = 1,
            kernel_cols: int = 1,
            kernel_depth: int = 1,
            stride_rows: int = 1,
            stride_cols: int = 1,
            stride_depth: int = 1,
            groups: int = 1,
            pad_top: int = 0,
            pad_right: int = 0,
            pad_front: int = 0,
            pad_bottom: int = 0,
            pad_left: int = 0,
            pad_back: int = 0,
            fine: int  = 1,
            input_t: FixedPoint = FixedPoint(16,8),
            output_t: FixedPoint = FixedPoint(16,8),
            weight_t: FixedPoint = FixedPoint(16,8),
            acc_t: FixedPoint = FixedPoint(32,16),
            has_bias: int = 0, # default to no bias for old configs
            block_floating_point: bool = False,
            backend: str = "chisel",
            regression_model: str = "linear_regression",
            stream_weights: int = 0,
            use_uram: bool = False,
            input_compression_ratio: list = [1.0],
            output_compression_ratio: list = [1.0],
            weight_compression_ratio: list = [1.0]
        ):

        # initialise parent class
        super().__init__(rows, cols, depth, channels,
                coarse_in, coarse_out, data_t=input_t,
                input_compression_ratio=input_compression_ratio,
                output_compression_ratio=output_compression_ratio)

        # save data types
        self.input_t = input_t
        self.output_t = output_t
        self.weight_t = weight_t
        self.acc_t = acc_t
        self.block_floating_point = block_floating_point

        # save bias flag
        self.has_bias = has_bias

        # init variables
        self._kernel_rows = kernel_rows
        self._kernel_cols = kernel_cols
        self._kernel_depth = kernel_depth
        self._stride_rows = stride_rows
        self._stride_cols = stride_cols
        self._stride_depth = stride_depth
        self._pad_top = pad_top
        self._pad_right = pad_right
        self._pad_front = pad_front
        self._pad_bottom = pad_bottom
        self._pad_left = pad_left
        self._pad_back = pad_back
        self._groups = groups
        self._coarse_group = coarse_group
        self._fine = fine
        self._filters = filters

        # check if the layer is depthwise
        self.depthwise = (groups == channels) and (groups == filters)

        # backend flag
        assert backend in ["hls", "chisel"], f"{backend} is an invalid backend"
        self.backend = backend

        # weights buffering flag
        if self.backend == "hls":
            self.double_buffered = False
            self.stream_weights = 0
            self.data_packing = False
            self.use_uram = False
        elif self.backend == "chisel":
            self.double_buffered = False
            self.stream_weights = stream_weights
            self.data_packing = True
            self.use_uram = use_uram

        # off chip weight streaming attributes
        self.weight_array_unit_depth = 0
        self.weight_array_unit_width = 0
        self.weight_compression_ratio = weight_compression_ratio

        # regression model
        assert regression_model in ["linear_regression", "xgboost", "xgboost-kernel"], f"{regression_model} is an invalid regression model"
        self.regression_model = regression_model

        if self.backend == "hls":

            self.modules["sliding_window3d"] = SlidingWindow3D(
                self.rows_in(),
                self.cols_in(),
                self.depth_in(),
                self.channels_in()//(self.coarse_in*self.coarse_group),
                self.kernel_rows, self.kernel_cols, self.kernel_depth,
                self.stride_rows, self.stride_cols, self.stride_depth,
                self.pad_top, self.pad_right, self.pad_front, self.pad_bottom, self.pad_left, self.pad_back, backend=self.backend,
                regression_model=self.regression_model)

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

            self.modules["vector_dot3d"] = VectorDot3D(
                    self.rows_out(), self.cols_out(), self.depth_out(),
                    (self.channels*self.kernel_rows*self.kernel_cols*self.kernel_depth)//(
                        self.fine*self.coarse_in*self.coarse_group),
                    self.filters//(self.coarse_out*self.groups), self.fine,
                    backend=self.backend, regression_model=self.regression_model)

            self.modules["accum3d"] = Accum3D(
                    self.rows_out(), self.cols_out(), self.depth_out(),
                    (self.kernel_rows*self.kernel_cols*self.kernel_depth*self.channels_in()
                        )//(self.fine*self.coarse_in*self.groups),
                    self.filters//(self.coarse_out*self.groups), 1,
                    backend=self.backend, regression_model=self.regression_model)

        self.modules["glue3d"] = Glue3D(
                self.rows_out(), self.cols_out(), self.depth_out(),
                1, int(self.filters/self.coarse_out), self.coarse_in,
                self.coarse_out, backend=self.backend,
                regression_model=self.regression_model) # TODO

        if self.has_bias:
            self.modules["bias3d"] = Bias3D(
                    self.rows_out(), self.cols_out(), self.depth_out(),
                    1, self.filters//self.streams_out(), backend=self.backend,
                    regression_model=self.regression_model) # TODO

        self.modules["shift_scale3d"] = ShiftScale3D(
                self.rows_out(), self.cols_out(), self.depth_out(),
                1, self.filters//(self.coarse_out*self.coarse_group))


        # update modules
        self.update()

    @property
    def kernel_size(self) -> List[int]:
        return [ self._kernel_rows, self._kernel_cols, self._kernel_depth ]

    @property
    def kernel_rows(self) -> int:
        return self._kernel_rows

    @property
    def kernel_cols(self) -> int:
        return self._kernel_cols

    @property
    def kernel_depth(self) -> int:
        return self._kernel_depth

    @property
    def stride(self) -> List[int]:
        return [ self._stride_rows, self._stride_cols, self._stride_depth ]

    @property
    def stride_rows(self) -> int:
        return self._stride_rows

    @property
    def stride_cols(self) -> int:
        return self._stride_cols

    @property
    def stride_depth(self) -> int:
        return self._stride_depth

    @property
    def pad(self) -> List[int]:
        return [
            self._pad_top,
            self._pad_left,
            self._pad_front,
            self._pad_bottom,
            self._pad_right,
            self._pad_back,
        ]

    @property
    def pad_top(self) -> int:
        return self._pad_top

    @property
    def pad_right(self) -> int:
        return self._pad_right

    @property
    def pad_front(self) -> int:
        return self._pad_front

    @property
    def pad_bottom(self) -> int:
        return self._pad_bottom

    @property
    def pad_left(self) -> int:
        return self._pad_left

    @property
    def pad_back(self) -> int:
        return self._pad_back

    @property
    def groups(self) -> int:
        return self._groups

    @property
    def coarse_group(self) -> int:
        return self._coarse_group

    @property
    def fine(self) -> int:
        return self._fine

    @property
    def filters(self) -> int:
        return self._filters

    @kernel_rows.setter
    def kernel_rows(self, val: int) -> None:
        self._kernel_rows = val
        # self.update()

    @kernel_cols.setter
    def kernel_cols(self, val: int) -> None:
        self._kernel_cols = val
        # self.update()

    @kernel_depth.setter
    def kernel_depth(self, val: int) -> None:
        self._kernel_depth = val
        # self.update()

    @stride_rows.setter
    def stride_rows(self, val: int) -> None:
        self._stride_rows = val
        # self.update()

    @stride_cols.setter
    def stride_cols(self, val: int) -> None:
        self._stride_cols = val
        # self.update()

    @stride_depth.setter
    def stride_depth(self, val: int) -> None:
        self._stride_depth = val
        # self.update()

    @pad_top.setter
    def pad_top(self, val: int) -> None:
        self._pad_top = val
        # self.update()

    @pad_right.setter
    def pad_right(self, val: int) -> None:
        self._pad_right = val
        # self.update()

    @pad_front.setter
    def pad_front(self, val: int) -> None:
        self._pad_front = val
        # self.update()

    @pad_bottom.setter
    def pad_bottom(self, val: int) -> None:
        self._pad_bottom = val
        # self.update()

    @pad_left.setter
    def pad_left(self, val: int) -> None:
        self._pad_left = val
        # self.update()

    @pad_back.setter
    def pad_back(self, val: int) -> None:
        self._pad_back = val
        # self.update()

    @groups.setter
    def groups(self, val: int) -> None:
        self._groups = val
        # self.update()

    @fine.setter
    def fine(self, val: int) -> None:
        self._fine = val
        # self.update()

    @filters.setter
    def filters(self, val: int) -> None:
        self._filters = val
        # self.update()

    @coarse_group.setter
    def coarse_group(self, val: int) -> None:
        assert(val in self.get_coarse_group_feasible())
        self._coarse_group = val
        # self.update()

    def start_depth(self):
        return (self.kernel_rows-1-self.pad_top)*self.cols*self.depth*self.channels//self.streams_in() + \
            (self.kernel_cols-1-self.pad_left)*self.depth*self.channels//self.streams_in() + \
            (self.kernel_depth-1-self.pad_front)*self.channels//self.streams_in() + \
            self.channels//self.streams_in()

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


    def update(self):
        if self.backend == "chisel":
            # pad
            self.modules['pad3d'].rows     = self.rows_in()
            self.modules['pad3d'].cols     = self.cols_in()
            self.modules['pad3d'].depth    = self.depth_in()
            self.modules['pad3d'].channels = self.channels_in()//(self.coarse_in*self.coarse_group)
            self.modules['pad3d'].data_width = self.input_t.width
            self.modules['pad3d'].pad_top = self.pad_top
            self.modules['pad3d'].pad_bottom = self.pad_bottom
            self.modules['pad3d'].pad_left = self.pad_left
            self.modules['pad3d'].pad_right = self.pad_right
            self.modules['pad3d'].pad_front = self.pad_front
            self.modules['pad3d'].pad_back = self.pad_back
            if self.data_packing:
                self.modules['pad3d'].streams = self.coarse_in*self.coarse_group

            # sliding window
            self.modules['sliding_window3d'].rows     = self.rows_in() + self.pad_top + self.pad_bottom
            self.modules['sliding_window3d'].cols     = self.cols_in() + self.pad_left + self.pad_right
            self.modules['sliding_window3d'].depth    = self.depth_in() + self.pad_front + self.pad_back
            self.modules['sliding_window3d'].channels = self.channels_in()//(self.coarse_in*self.coarse_group)
            self.modules['sliding_window3d'].kernel_cols = self.kernel_cols
            self.modules['sliding_window3d'].kernel_rows = self.kernel_rows
            self.modules['sliding_window3d'].kernel_depth= self.kernel_depth
            self.modules['sliding_window3d'].stride_cols = self.stride_cols
            self.modules['sliding_window3d'].stride_rows = self.stride_rows
            self.modules['sliding_window3d'].stride_depth= self.stride_depth
            self.modules['sliding_window3d'].data_width = self.input_t.width
            self.modules['sliding_window3d'].pad_top = 0
            self.modules['sliding_window3d'].pad_bottom = 0
            self.modules['sliding_window3d'].pad_left = 0
            self.modules['sliding_window3d'].pad_right = 0
            self.modules['sliding_window3d'].pad_front = 0
            self.modules['sliding_window3d'].pad_back = 0
            if self.data_packing:
                self.modules['sliding_window3d'].streams = self.coarse_in*self.coarse_group

            # squeeze3d
            self.modules['squeeze3d'].rows     = self.rows_out()
            self.modules['squeeze3d'].cols     = self.cols_out()
            self.modules['squeeze3d'].depth    = self.depth_out()
            self.modules['squeeze3d'].channels = self.channels//(self.coarse_in*self.coarse_group)
            self.modules['squeeze3d'].coarse_out = self.fine
            # self.modules['squeeze3d'].coarse_in  = self.kernel_rows*self.kernel_cols*self.kernel_depth
            self.modules['squeeze3d'].data_width = self.input_t.width
            if self.data_packing:
                self.modules['squeeze3d'].streams = self.coarse_in*self.coarse_group

            elif self.backend == "hls":
                # sliding window
                self.modules['sliding_window3d'].rows     = self.rows
                self.modules['sliding_window3d'].cols     = self.cols
                self.modules['sliding_window3d'].depth    = self.depth
                self.modules['sliding_window3d'].channels = self.channels//(self.coarse_in*self.coarse_group)
                self.modules['sliding_window3d'].data_width   = self.input_t.width
                if self.data_packing:
                    self.modules['sliding_window3d'].streams = self.coarse_in*self.coarse_group

        # fork3d
        self.modules['fork3d'].rows     = self.rows_out()
        self.modules['fork3d'].cols     = self.cols_out()
        self.modules['fork3d'].depth    = self.depth_out()
        self.modules['fork3d'].channels = self.channels_in()//(self.coarse_in*self.coarse_group)
        self.modules['fork3d'].coarse   = self.coarse_out
        self.modules['fork3d'].data_width     = self.input_t.width
        if self.backend == "chisel":
            self.modules['fork3d'].kernel_rows  = self.fine
            self.modules['fork3d'].kernel_cols  = 1
            self.modules['fork3d'].kernel_depth = 1
        if self.data_packing:
            self.modules['fork3d'].streams = self.coarse_in*self.coarse_group

        if self.backend == "hls":
            # TODO: check the group parameter
            self.modules['conv3d'].rows     = self.rows_out()
            self.modules['conv3d'].cols     = self.cols_out()
            self.modules['conv3d'].depth    = self.depth_out()
            self.modules['conv3d'].channels = self.channels_in()//(self.coarse_in*self.coarse_group)
            self.modules['conv3d'].filters  = self.filters//(self.coarse_out*self.coarse_group)
            self.modules['conv3d'].groups   = self.groups // self.coarse_group
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
        if self.backend == "hls":
            self.modules['accum3d'].filters  = self.filters//(self.coarse_out*self.coarse_group)
            self.modules['accum3d'].channels  = self.channels_in()//(self.coarse_in*self.coarse_group)
            self.modules['accum3d'].groups   = self.groups//self.coarse_group
        elif self.backend == "chisel":
            self.modules['accum3d'].filters  = self.filters//(self.coarse_out*self.groups)
            self.modules['accum3d'].groups   = 1
            self.modules['accum3d'].channels = (
                    self.channels*self.kernel_rows*self.kernel_cols*self.kernel_depth)//(
                    self.fine*self.coarse_in*self.coarse_group)
        if self.data_packing:
            self.modules['accum3d'].streams = self.coarse_in*self.coarse_group*self.coarse_out

        # glue3d
        self.modules['glue3d'].rows       = self.rows_out()
        self.modules['glue3d'].cols       = self.cols_out()
        self.modules['glue3d'].depth      = self.depth_out()
        self.modules['glue3d'].filters    = self.filters//self.coarse_group
        self.modules['glue3d'].coarse_in  = self.coarse_in
        self.modules['glue3d'].coarse_out = self.coarse_out
        self.modules['glue3d'].coarse_group = self.coarse_group
        self.modules['glue3d'].data_width = self.acc_t.width
        if self.data_packing:
            self.modules['glue3d'].streams = self.coarse_group*self.coarse_out

        if self.has_bias:
            # bias3d
            self.modules['bias3d'].rows           = self.rows_out()
            self.modules['bias3d'].cols           = self.cols_out()
            self.modules['bias3d'].depth          = self.depth_out()
            self.modules['bias3d'].filters        = self.filters//(self.coarse_group*self.coarse_out)
            self.modules['bias3d'].data_width     = self.output_t.width
            self.modules['bias3d'].biases_width   = self.acc_t.width
            if self.data_packing:
                self.modules['bias3d'].streams = self.coarse_out*self.coarse_group

        self.modules['shift_scale3d'].rows           = self.rows_out()
        self.modules['shift_scale3d'].cols           = self.cols_out()
        self.modules['shift_scale3d'].depth          = self.depth_out()
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
        self.input_t.to_protobuf(parameters.input_t)
        self.output_t.to_protobuf(parameters.output_t)
        self.weight_t.to_protobuf(parameters.weight_t)
        self.acc_t.to_protobuf(parameters.acc_t)
        parameters.data_t.Clear()
        parameters.use_uram     = self.use_uram
        if self.weights_ram_usage + self.stream_weights > 0:
            parameters.on_chip_addr_range = int(self.on_chip_addr_range())
        else:
            parameters.on_chip_addr_range = 0
        parameters.stream_weights = int(self.stream_weights)
        if self.stream_weights > 0:
            parameters.off_chip_buffer_size = self.off_chip_buffer_size()
            parameters.off_chip_interval = math.ceil(self.on_chip_addr_range() / (self.stream_weights / self.stream_unit()))
        else:
            parameters.off_chip_buffer_size = 0
            parameters.off_chip_interval = -1
        parameters.weight_compression_ratio.extend(self.weight_compression_ratio)

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
        weights_size =  (self.channels_in() * self.filters) // self.groups \
             * self.kernel_rows * self.kernel_cols * self.kernel_depth
        if self.has_bias:
            bias_size = self.filters
        else:
            bias_size = 0
        return {
            "weights"   : weights_size,
            "bias"      : bias_size
        }

    def get_operations(self):
        # 1 MAC = 2 OPs
        ops = 2* (self.channels_in() * self.filters) // self.groups \
            * self.kernel_rows * self.kernel_cols * self.kernel_depth \
            * self.rows_out() * self.cols_out() * self.depth_out()
        if self.has_bias:
            ops += self.filters \
                * self.rows_out() * self.cols_out() * self.depth_out()
        return ops

    def resource(self):

        if self.backend == "chisel":

            # get module resource models
            pad_rsc         = self.modules['pad3d'].rsc()
            sw_rsc          = self.modules['sliding_window3d'].rsc()
            squeeze_rsc     = self.modules['squeeze3d'].rsc()
            fork_rsc        = self.modules['fork3d'].rsc()
            vector_dot_rsc  = self.modules['vector_dot3d'].rsc()
            accum_rsc       = self.modules['accum3d'].rsc()
            glue_rsc        = self.modules['glue3d'].rsc()
            if self.has_bias:
                bias_rsc        = self.modules['bias3d'].rsc()
            shift_scale_rsc = self.modules['shift_scale3d'].rsc()

            self.modules['sliding_window3d'].buffer_estimate()
            line_buffer_bram = self.modules['sliding_window3d'].line_buffer_bram
            if self.stream_inputs[0]:
                sw_rsc["BRAM"] -= line_buffer_bram
                self.inputs_ram_usage = [0]
            else:
                self.inputs_ram_usage = [line_buffer_bram]

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

            # dsp packing
            if self.weight_t.width <= 4 and self.input_t.width <= 4:
                vector_dot_rsc["DSP"] = vector_dot_rsc["DSP"]*0.25
            elif self.weight_t.width <= 8 and self.input_t.width <= 8:
                vector_dot_rsc["DSP"] = vector_dot_rsc["DSP"]*0.5

            if self.data_packing:
                rsc = { rsc_type: (
                    pad_rsc[rsc_type] +
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
                # accumulate resource usage based on coarse factors
                rsc = { rsc_type: (
                    pad_rsc[rsc_type]*self.coarse_in*self.coarse_group +
                    sw_rsc[rsc_type]*self.coarse_in*self.coarse_group +
                    squeeze_rsc[rsc_type]*self.coarse_in*self.coarse_group +
                    fork_rsc[rsc_type]*self.coarse_in*self.coarse_group +
                    math.ceil(vector_dot_rsc[rsc_type]*self.coarse_in*self.coarse_out*self.coarse_group) +
                    accum_rsc[rsc_type]*self.coarse_in*self.coarse_out*self.coarse_group +
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
            if self.has_bias:
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

        # apply data packing
        if self.data_packing:
            weight_array_depth = math.ceil(weight_memory_depth)
            weight_array_width = self.weight_t.width*self.fine*self.coarse_in*self.coarse_out*self.coarse_group
            weight_array_num = 1
        else:
            weight_array_depth = math.ceil(weight_memory_depth)
            weight_array_width = self.weight_t.width
            weight_array_num = self.fine*self.coarse_in*self.coarse_out*self.coarse_group
        weights_bram_usage, weights_uram_usage = self.stream_rsc(weight_array_depth, weight_array_width, weight_array_num)

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

        # add weight, bias, shift_scale to resources
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
        if self.has_bias:
            assert bias.shape[0] == self.filters  ,     "ERROR (bias): invalid filter dimension"

        # instantiate convolution layer
        # convolution_layer = torch.nn.Conv3d(self.channels_in(), self.filters, (self.kernel_depth, self.kernel_rows, self.kernel_cols), stride=(self.stride_depth, self.stride_rows, self.stride_cols), padding=(self.pad_front, self.pad_top, self.pad_right), groups=self.groups, bias=True)
        convolution_layer = torch.nn.Conv3d(self.channels_in(), self.filters, (self.kernel_depth, self.kernel_rows, self.kernel_cols), stride=(self.stride_depth, self.stride_rows, self.stride_cols), padding=0, groups=self.groups, bias=True)

        # update weights
        convolution_layer.weight = torch.nn.Parameter(
                torch.from_numpy(np.moveaxis(weights,-1,-3)))

        # update bias
        if self.has_bias:
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

