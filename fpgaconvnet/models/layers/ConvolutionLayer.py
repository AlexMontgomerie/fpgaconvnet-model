import importlib
import math
from typing import Union, List

import pydot
import numpy as np

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model
from fpgaconvnet.models.layers import Layer

from fpgaconvnet.models.modules import Pad
from fpgaconvnet.models.modules import SlidingWindow
from fpgaconvnet.models.modules import VectorDot
from fpgaconvnet.models.modules import Conv
from fpgaconvnet.models.modules import Squeeze
from fpgaconvnet.models.modules import Fork
from fpgaconvnet.models.modules import Accum
from fpgaconvnet.models.modules import Glue
from fpgaconvnet.models.modules import Bias
from fpgaconvnet.models.modules import ShiftScale


class ConvolutionLayer(Layer):

    def __init__(
            self,
            filters: int,
            rows: int,
            cols: int,
            channels: int,
            coarse_in: int = 1,
            coarse_out: int = 1,
            coarse_group: int = 1,
            kernel_rows: int = 1,
            kernel_cols: int = 1,
            stride_rows: int = 1,
            stride_cols: int = 1,
            groups: int = 1,
            pad_top: int = 0,
            pad_right: int = 0,
            pad_bottom: int = 0,
            pad_left: int = 0,
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
        super().__init__(rows, cols, channels,
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
        self._stride_rows = stride_rows
        self._stride_cols = stride_cols
        self._pad_top       = pad_top
        self._pad_right     = pad_right
        self._pad_bottom    = pad_bottom
        self._pad_left      = pad_left
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
        assert regression_model in ["linear_regression", "xgboost"], f"{regression_model} is an invalid regression model"
        self.regression_model = regression_model

        if self.backend == "hls":
            self.modules["sliding_window"] = SlidingWindow(self.rows_in(), self.cols_in(),
                    self.channels_in()//(self.coarse_in*self.coarse_group), self.kernel_size,
                    self.stride, self.pad_top, self.pad_right, self.pad_bottom, self.pad_left,
                    backend=self.backend, regression_model=self.regression_model)

            self.modules["fork"] = Fork(self.rows_out(), self.cols_out(),
                    self.channels_in()//(self.coarse_in*self.coarse_group),
                    self.kernel_size, self.coarse_out, backend=self.backend, regression_model=self.regression_model)

            self.modules["conv"] = Conv(self.rows_out(), self.cols_out(),
                    self.channels_in()//(self.coarse_in*self.coarse_group),
                    self.filters//(self.coarse_out*self.coarse_group),
                    self.fine, self.kernel_size,
                    self.groups//self.coarse_group,
                    backend=self.backend, regression_model=self.regression_model)

            self.modules["accum"] = Accum(self.rows_out(), self.cols_out(),
                    self.channels_in()//(self.coarse_in*self.coarse_group),
                    self.filters//(self.coarse_out*self.coarse_group),
                    self.groups//self.coarse_group,
                    backend=self.backend, regression_model=self.regression_model)

        elif self.backend == "chisel":
            self.modules["pad"] = Pad(
                self.rows_in(), self.cols_in(), self.channels_in()//(self.coarse_in*self.coarse_group),
                self.pad_top, self.pad_bottom, self.pad_left, self.pad_right, backend=self.backend,
                regression_model=self.regression_model)

            self.modules["sliding_window"] = SlidingWindow(
                self.rows_in() + self.pad_top + self.pad_bottom,
                self.cols_in() + self.pad_left + self.pad_right,
                self.channels_in()//(self.coarse_in*self.coarse_group),
                self.kernel_size,
                self.stride,
                0, 0, 0, 0, backend=self.backend,
                regression_model=self.regression_model)

            self.modules["squeeze"] = Squeeze(self.rows_out(), self.cols_out(),
                    self.channels_in()//(self.coarse_in*self.coarse_group),
                    self.kernel_size[0]*self.kernel_size[1], self.fine,
                    backend=self.backend, regression_model=self.regression_model)

            self.modules["fork"] = Fork(self.rows_out(), self.cols_out(),
                    self.channels_in()//(self.coarse_in*self.coarse_group),
                    [self.fine, 1], self.coarse_out, backend=self.backend, regression_model=self.regression_model)

            self.modules["vector_dot"] = VectorDot(self.rows_out(), self.cols_out(),
                    (self.channels*self.kernel_size[0]*self.kernel_size[1])//(
                        self.fine*self.coarse_in*self.coarse_group),
                    self.filters//(self.coarse_out*self.groups), self.fine,
                    backend=self.backend, regression_model=self.regression_model)

            self.modules["accum"] = Accum(self.rows_out(), self.cols_out(),
                    (self.kernel_size[0]*self.kernel_size[1]*self.channels_in())//(
                        self.fine*self.coarse_in*self.coarse_group),
                    self.filters//(self.coarse_out*self.groups), 1,
                    backend=self.backend, regression_model=self.regression_model)

        self.modules["glue"] = Glue(self.rows_out(), self.cols_out(), 1,
                int(self.filters/self.coarse_out), self.coarse_in, self.coarse_out, self.coarse_group,
                backend=self.backend, regression_model=self.regression_model)

        if self.has_bias:
            self.modules["bias"] = Bias(self.rows_out(), self.cols_out(), 1, self.filters//self.streams_out(), backend=self.backend, regression_model=self.regression_model)

        self.modules["shift_scale"] = ShiftScale(self.rows_out(), self.cols_out(), 1, self.filters//(self.coarse_out*self.coarse_group),
                backend=self.backend, regression_model=self.regression_model)

        # update modules
        self.update()

    @property
    def kernel_size(self) -> List[int]:
        return [ self._kernel_rows, self._kernel_cols ]

    @property
    def kernel_rows(self) -> int:
        return self._kernel_rows

    @property
    def kernel_cols(self) -> int:
        return self._kernel_cols

    @property
    def stride(self) -> List[int]:
        return [ self._stride_rows, self._stride_cols ]

    @property
    def stride_rows(self) -> int:
        return self._stride_rows

    @property
    def stride_cols(self) -> int:
        return self._stride_cols

    @property
    def pad(self) -> List[int]:
        return [
            self._pad_top,
            self._pad_left,
            self._pad_bottom,
            self._pad_right,
        ]

    @property
    def pad_top(self) -> int:
        return self._pad_top

    @property
    def pad_right(self) -> int:
        return self._pad_right

    @property
    def pad_bottom(self) -> int:
        return self._pad_bottom

    @property
    def pad_left(self) -> int:
        return self._pad_left

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

    @kernel_size.setter
    def kernel_size(self, val: List[int]) -> None:
        self._kernel_rows = val[0]
        self._kernel_cols = val[1]
        # self.update()

    @kernel_rows.setter
    def kernel_rows(self, val: int) -> None:
        self._kernel_rows = val
        # self.update()

    @kernel_cols.setter
    def kernel_cols(self, val: int) -> None:
        self._kernel_cols = val
        # self.update()

    @stride.setter
    def stride(self, val: List[int]) -> None:
        self._stride_rows = val[0]
        self._stride_cols = val[1]
        # self.update()

    @stride_rows.setter
    def stride_rows(self, val: int) -> None:
        self._stride_rows = val
        # self.update()

    @stride_cols.setter
    def stride_cols(self, val: int) -> None:
        self._stride_cols = val
        # self.update()

    @pad.setter
    def pad(self, val: List[int]) -> None:
        self._pad_top    = val[0]
        self._pad_right  = val[3]
        self._pad_bottom = val[2]
        self._pad_left   = val[1]
        # self.update()

    @pad_top.setter
    def pad_top(self, val: int) -> None:
        self._pad_top = val
        # self.update()

    @pad_right.setter
    def pad_right(self, val: int) -> None:
        self._pad_right = val
        # self.update()

    @pad_bottom.setter
    def pad_bottom(self, val: int) -> None:
        self._pad_bottom = val
        # self.update()

    @pad_left.setter
    def pad_left(self, val: int) -> None:
        self._pad_left = val
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

    @Layer.coarse_in.setter
    def coarse_in(self, val: int) -> None:
        assert(val in self.get_coarse_in_feasible())
        self._coarse_in = val
        self.update()

    @coarse_group.setter
    def coarse_group(self, val: int) -> None:
        assert(val in self.get_coarse_group_feasible())
        self._coarse_group = val
        self.update()

    def rows_out(self) -> int:
        return self.modules["sliding_window"].rows_out()

    def cols_out(self) -> int:
        return self.modules["sliding_window"].cols_out()

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

    def start_depth(self):
        return (self.kernel_rows-1-self.pad_top)*self.cols*self.channels//self.streams_in() + \
               (self.kernel_cols-1-self.pad_left)*self.channels//self.streams_in() + \
                self.channels//self.streams_in()

    def update(self):
        if self.backend == "chisel":
            # pad
            self.modules['pad'].rows     = self.rows_in()
            self.modules['pad'].cols     = self.cols_in()
            self.modules['pad'].channels = self.channels_in()//(self.coarse_in*self.coarse_group)
            self.modules['pad'].data_width = self.data_t.width
            self.modules['pad'].pad_top = self.pad_top
            self.modules['pad'].pad_bottom = self.pad_bottom
            self.modules['pad'].pad_left = self.pad_left
            self.modules['pad'].pad_right = self.pad_right
            if self.data_packing:
                self.modules['pad'].streams = self.coarse_in*self.coarse_group
            else:
                self.modules['pad'].streams = 1

            # sliding window
            self.modules['sliding_window'].rows     = self.rows_in() + self.pad_top + self.pad_bottom
            self.modules['sliding_window'].cols     = self.cols_in() + self.pad_left + self.pad_right
            self.modules['sliding_window'].channels = self.channels_in()//(self.coarse_in*self.coarse_group)
            self.modules['sliding_window'].kernel_cols = self.kernel_cols
            self.modules['sliding_window'].kernel_rows = self.kernel_rows
            self.modules['sliding_window'].stride_cols = self.stride_cols
            self.modules['sliding_window'].stride_rows = self.stride_rows
            self.modules['sliding_window'].data_width = self.data_t.width
            self.modules['sliding_window'].pad_top = 0
            self.modules['sliding_window'].pad_bottom = 0
            self.modules['sliding_window'].pad_left = 0
            self.modules['sliding_window'].pad_right = 0
            if self.data_packing:
                self.modules['sliding_window'].streams = self.coarse_in*self.coarse_group
            else:
                self.modules['sliding_window'].streams = 1

            # squeeze
            self.modules['squeeze'].rows     = self.rows_out()
            self.modules['squeeze'].cols     = self.cols_out()
            self.modules['squeeze'].channels = self.channels//(self.coarse_in*self.coarse_group)
            self.modules['squeeze'].coarse_out = self.fine
            self.modules['squeeze'].data_width = self.input_t.width
            if self.data_packing:
                self.modules['squeeze'].streams = self.coarse_in*self.coarse_group
            else:
                self.modules['squeeze'].streams = 1

        elif self.backend == "hls":
            # sliding window
            self.modules['sliding_window'].rows     = self.rows
            self.modules['sliding_window'].cols     = self.cols
            self.modules['sliding_window'].channels = self.channels//(self.coarse_in*self.coarse_group)
            self.modules['sliding_window'].data_width   = self.input_t.width
            if self.data_packing:
                self.modules['sliding_window'].streams = self.coarse_in*self.coarse_group
            else:
                self.modules['sliding_window'].streams = 1

        # fork
        self.modules['fork'].rows     = self.rows_out()
        self.modules['fork'].cols     = self.cols_out()
        self.modules['fork'].channels = self.channels_in()//(self.coarse_in*self.coarse_group)
        self.modules['fork'].coarse   = self.coarse_out
        self.modules['fork'].data_width     = self.input_t.width
        if self.backend == "chisel":
            self.modules['fork'].kernel_size = [self.fine, 1]
        if self.data_packing:
            self.modules['fork'].streams = self.coarse_in*self.coarse_group
        else:
            self.modules['fork'].streams = 1

        if self.backend == "hls":
            # TODO: check the group parameter
            self.modules['conv'].rows     = self.rows_out()
            self.modules['conv'].cols     = self.cols_out()
            self.modules['conv'].channels = self.channels_in()//(self.coarse_in*self.coarse_group)
            self.modules['conv'].filters  = self.filters//(self.coarse_out*self.coarse_group)
            self.modules['conv'].groups = self.groups // self.coarse_group
            self.modules['conv'].fine     = self.fine
            self.modules['conv'].data_width     = self.input_t.width
            self.modules['conv'].weight_width   = self.weight_t.width
            self.modules['conv'].acc_width      = self.acc_t.width
        elif self.backend == "chisel":
            # kernel dot
            self.modules['vector_dot'].rows     = self.rows_out()
            self.modules['vector_dot'].cols     = self.cols_out()
            self.modules['vector_dot'].filters  = self.filters//(self.coarse_out*self.groups)
            self.modules['vector_dot'].data_width     = self.input_t.width
            self.modules['vector_dot'].weight_width   = self.weight_t.width
            self.modules['vector_dot'].acc_width      = self.acc_t.width
            self.modules['vector_dot'].fine     = self.fine

            self.modules['vector_dot'].channels = (
                self.channels*self.kernel_size[0]*self.kernel_size[1])//(
                self.fine*self.coarse_in*self.coarse_group)

            if self.data_packing:
                self.modules['vector_dot'].streams = self.coarse_in*self.coarse_out*self.coarse_group
            else:
                self.modules['vector_dot'].streams = 1

        # accum
        self.modules['accum'].rows     = self.rows_out()
        self.modules['accum'].cols     = self.cols_out()
        self.modules['accum'].data_width    = self.acc_t.width
        if self.backend == "hls":
            self.modules['accum'].filters  = self.filters//(self.coarse_out*self.coarse_group)
            self.modules['accum'].channels  = self.channels_in()//(self.coarse_in*self.coarse_group)
            self.modules['accum'].groups   = self.groups//self.coarse_group
        elif self.backend == "chisel":
            self.modules['accum'].filters  = self.filters//(self.coarse_out*self.groups)
            self.modules['accum'].groups   = 1
            self.modules['accum'].channels = (
                self.channels*self.kernel_size[0]*self.kernel_size[1])//(
                self.fine*self.coarse_in*self.coarse_group)
        if self.data_packing:
            self.modules['accum'].streams = self.coarse_in*self.coarse_group*self.coarse_out
        else:
            self.modules['accum'].streams = 1

        # glue
        self.modules['glue'].rows       = self.rows_out()
        self.modules['glue'].cols       = self.cols_out()
        self.modules['glue'].filters    = self.filters//self.coarse_group
        self.modules['glue'].coarse_in  = self.coarse_in
        self.modules['glue'].coarse_out = self.coarse_out
        self.modules['glue'].coarse_group = self.coarse_group
        self.modules['glue'].data_width = self.acc_t.width
        if self.data_packing:
            self.modules['glue'].streams = self.coarse_group*self.coarse_out
        else:
            self.modules['glue'].streams = 1

        if self.has_bias:
            # bias
            self.modules['bias'].rows           = self.rows_out()
            self.modules['bias'].cols           = self.cols_out()
            self.modules['bias'].filters        = self.filters//(self.coarse_out*self.coarse_group)
            self.modules['bias'].data_width     = self.output_t.width
            self.modules['bias'].biases_width   = self.acc_t.width
            if self.data_packing:
                self.modules['bias'].streams = self.coarse_out*self.coarse_group
            else:
                self.modules['bias'].streams = 1

        # shift scale
        self.modules['shift_scale'].rows           = self.rows_out()
        self.modules['shift_scale'].cols           = self.cols_out()
        self.modules['shift_scale'].filters        = self.filters//(self.coarse_out*self.coarse_group)
        self.modules['shift_scale'].data_width     = self.output_t.width
        self.modules['shift_scale'].biases_width   = self.acc_t.width
        if self.data_packing:
            self.modules['shift_scale'].streams = self.coarse_out*self.coarse_group
        else:
            self.modules['shift_scale'].streams = 1

    def layer_info(self,parameters,batch_size=1):
        Layer.layer_info(self, parameters, batch_size)
        parameters.filters      = self.filters
        parameters.groups       = self.groups
        parameters.coarse_group = self.coarse_group
        parameters.kernel_size.extend(self.kernel_size)
        parameters.kernel_rows  = self.kernel_rows
        parameters.kernel_cols  = self.kernel_cols
        parameters.stride.extend(self.stride)
        parameters.stride_rows  = self.stride_rows
        parameters.stride_cols  = self.stride_cols
        parameters.pad_top      = self.pad_top
        parameters.pad_right    = self.pad_right
        parameters.pad_bottom   = self.pad_bottom
        parameters.pad_left     = self.pad_left
        parameters.has_bias     = self.has_bias
        parameters.fine  = self.fine
        parameters.use_uram     = self.use_uram
        parameters.block_floating_point    = self.block_floating_point
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
            return get_factors(self.kernel_size[0]*self.kernel_size[1])
        elif self.backend == "hls":
            if self.kernel_size[0] != self.kernel_size[1]:
                # assert(self.kernel_size[0] == 1 or self.kernel_size[1] == 1)
                return [ 1, max(self.kernel_size[0],self.kernel_size[1])]
            else:
                return [ 1, self.kernel_size[0], self.kernel_size[0]*self.kernel_size[1] ]

    def get_weights_reloading_feasible(self):
        return get_factors(self.filters//(self.groups*self.coarse_out))

    def get_parameters_size(self):
        weights_size = (self.channels_in() * self.filters) // self.groups \
                * self.kernel_size[0] * self.kernel_size[1]
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
        ops = 2 * (self.channels_in() * self.filters) // self.groups \
                * self.kernel_size[0] * self.kernel_size[1] \
                * self.rows_out() * self.cols_out()
        if self.has_bias:
            ops += self.filters * self.rows_out() * self.cols_out()
        return ops

    def resource(self):

        if self.backend == "chisel":

            # get module resource models
            pad_rsc         = self.modules['pad'].rsc()
            sw_rsc          = self.modules['sliding_window'].rsc()
            squeeze_rsc     = self.modules['squeeze'].rsc()
            fork_rsc        = self.modules['fork'].rsc()
            vector_dot_rsc  = self.modules['vector_dot'].rsc()
            accum_rsc       = self.modules['accum'].rsc()
            glue_rsc        = self.modules['glue'].rsc()
            if self.has_bias:
                bias_rsc        = self.modules['bias'].rsc()
            shift_scale_rsc = self.modules['shift_scale'].rsc()

            # for streamed inputs, the line buffer is skipped
            self.modules['sliding_window'].buffer_estimate()
            line_buffer_bram = self.modules['sliding_window'].line_buffer_bram
            if self.stream_inputs[0]:
                sw_rsc["BRAM"] -= line_buffer_bram
                self.inputs_ram_usage = [0]
            else:
                self.inputs_ram_usage = [line_buffer_bram]

            # remove redundant modules
            if self.kernel_size[0] == 1 and self.kernel_size[1] == 1:
                sw_rsc      = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            if self.fine == self.kernel_size[0]*self.kernel_size[1]:
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
            sw_rsc      = self.modules['sliding_window'].rsc()
            fork_rsc    = self.modules['fork'].rsc()
            conv_rsc    = self.modules['conv'].rsc()
            accum_rsc   = self.modules['accum'].rsc()
            glue_rsc    = self.modules['glue'].rsc()
            if self.has_bias:
                bias_rsc    = self.modules['bias'].rsc()

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
                                    self.kernel_size[0]* \
                                    self.kernel_size[1]) / \
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
        pass
        """
        cluster = pydot.Cluster(name, label=name,
                style="dashed", bgcolor="lightpink")

        # names
        slwin_name = [[""]*self.coarse_in]*self.coarse_group
        fork_name = [[""]*self.coarse_in]*self.coarse_group
        conv_name = [[[""]*self.coarse_in]*self.coarse_out]*self.coarse_group
        accum_name = [[[""]*self.coarse_in]*self.coarse_out]*self.coarse_group
        glue_name = [[""]*self.coarse_out]*self.coarse_group
        bias_name = [[""]*self.coarse_out]*self.coarse_group

        for g in range(self.coarse_group):
            for i in range(self.coarse_in):
                # define names
                slwin_name[g][i] = "_".join([name, "sw", str(g), str(i)])
                fork_name[g][i] = "_".join([name, "fork", str(g), str(i)])
                # add nodes
                cluster.add_node(self.modules["sliding_window"].visualise(slwin_name[g][i]))
                cluster.add_node(self.modules["fork"].visualise(fork_name[g][i]))
                # add edges
                cluster.add_edge(pydot.Edge(slwin_name[g][i], fork_name[g][i]))

                # iterate over coarse out
                for j in range(self.coarse_out):
                    # define names
                    conv_name[g][j][i] = "_".join([name, "conv", str(g), str(j), str(i)])
                    accum_name[g][j][i] = "_".join([name, "accum", str(g), str(j), str(i)])
                    glue_name[g][j] = "_".join([name, "glue", str(g), str(j)])
                    bias_name[g][j] = "_".join([name, "bias", str(g), str(j)])

                    # add nodes
                    cluster.add_node(self.modules["conv"].visualise(conv_name[g][j][i]))
                    cluster.add_node(self.modules["accum"].visualise(accum_name[g][j][i]))

                    # add edges
                    cluster.add_edge(pydot.Edge(fork_name[g][i], conv_name[g][j][i]))
                    cluster.add_edge(pydot.Edge(conv_name[g][j][i], accum_name[g][j][i]))
                    cluster.add_edge(pydot.Edge(accum_name[g][j][i], glue_name[g][j]))

        for g in range(self.coarse_group):
            for j in range(self.coarse_out):

                # add nodes
                cluster.add_node(self.modules["glue"].visualise(glue_name[g][j]))
                cluster.add_node(self.modules["bias"].visualise(bias_name[g][j]))

                # add edges
                cluster.add_edge(pydot.Edge(glue_name[g][j], bias_name[g][j]))


        return cluster, np.array(slwin_name).flatten().tolist(), np.array(bias_name).flatten().tolist()
        """

    def functional_model(self,data,weights,bias,batch_size=1):
        import torch

        assert data.shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels_in(), "ERROR (data): invalid channel dimension"

        assert weights.shape[0] == self.filters ,   "ERROR (weights): invalid filter dimension"
        assert weights.shape[1] == self.channels//self.groups,\
                                                    "ERROR (weights): invalid channel dimension"
        assert weights.shape[2] == self.kernel_size[0],\
                                                    "ERROR (weights): invalid kernel dimension"
        assert weights.shape[3] == self.kernel_size[1],\
                                                    "ERROR (weights): invalid kernel dimension"
        if self.has_bias:
            assert bias.shape[0] == self.filters  ,     "ERROR (bias): invalid filter dimension"

        # instantiate convolution layer
        convolution_layer = torch.nn.Conv2d(self.channels_in(), self.filters, self.kernel_size,
                stride=self.stride, padding=0, groups=self.groups, bias=self.has_bias)

        # update weights
        convolution_layer.weight = torch.nn.Parameter(torch.from_numpy(weights))

        # update bias
        if self.has_bias:
            convolution_layer.bias = torch.nn.Parameter(torch.from_numpy(bias))

        # get the padding
        padding = [
            self.pad_left,
            self.pad_right,
            self.pad_top,
            self.pad_bottom
        ]

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        data = torch.nn.functional.pad(torch.from_numpy(data), padding, "constant", 0.0)
        data_out = convolution_layer(data).detach().numpy()

        return data_out
