import importlib
import math
from typing import Union, List

import pydot
import numpy as np
import torch

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model
from fpgaconvnet.models.layers import Layer

from fpgaconvnet.models.modules import SlidingWindow
from fpgaconvnet.models.modules import VectorDot
from fpgaconvnet.models.modules import SparseVectorDot
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
            stride_rows: int = 2,
            stride_cols: int = 2,
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
            sparsity: list = [],
            skipping_windows: bool = False,
            block_floating_point: bool = False,
            backend: str = "chisel", # default to no bias for old configs
            regression_model: str = "linear_regression"
        ):

        # initialise parent class
        super().__init__(rows, cols, channels,
                coarse_in, coarse_out, data_t=input_t)

        # save data types
        self.input_t = input_t
        self.output_t = output_t
        self.weight_t = weight_t
        self.acc_t = acc_t
        self.block_floating_point = block_floating_point

        # save bias flag
        self.has_bias = has_bias
        self.skipping_windows = skipping_windows

        self.window_sparsity = []
        self.sparsity = sparsity
        # save sparsity
        if len(sparsity) > 0:
            #Ensure histogram data provided
            assert (len(sparsity) == channels*(kernel_cols*kernel_rows+1))
            # reject if pointwise or low sparsity
            self.sparsity = np.array(sparsity).reshape((channels, kernel_rows*kernel_cols+1))
            self.window_sparsity = np.copy(np.squeeze(self.sparsity[:, -1]))
            weights = np.arange(self.sparsity.shape[1])
            avg_sparsity = np.sum(weights * self.sparsity, axis = 1)/(self.sparsity.shape[1] - 1)
            if kernel_rows == 1 and kernel_cols == 1 or np.mean(avg_sparsity) < 0.1:
                    self.skipping_windows = False
                    self.window_sparsity = []
                    self.sparsity = []
        else:
            self.skipping_windows = False
            self.window_sparsity = []

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
            self.stream_weights = False
            self.data_packing = False
            self.use_uram = False
        elif self.backend == "chisel":
            self.double_buffered = False
            self.stream_weights = False
            self.data_packing = True
            self.use_uram = False

        # regression model
        assert regression_model in ["linear_regression", "xgboost"], f"{regression_model} is an invalid regression model"
        self.regression_model = regression_model

        self.modules["sliding_window"] = SlidingWindow(self.rows_in(), self.cols_in(),
                self.channels_in()//(self.coarse_in*self.coarse_group), self.kernel_size,
                self.stride, self.pad_top, self.pad_right, self.pad_bottom, self.pad_left,
                backend=self.backend, regression_model=self.regression_model)

        if self.backend == "hls":

            self.modules["fork"] = Fork(self.rows_out(), self.cols_out(),
                    self.channels_in()//(self.coarse_in*self.coarse_group),
                    self.kernel_size, self.coarse_out, backend=self.backend, regression_model=self.regression_model)

            self.modules["Conv"] = Conv(self.rows_out(), self.cols_out(),
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

            self.modules["squeeze"] = Squeeze(self.rows_out(), self.cols_out(),
                    self.channels_in()//(self.coarse_in*self.coarse_group),
                    self.kernel_size[0]*self.kernel_size[1], self.fine,
                    backend=self.backend, regression_model=self.regression_model)

            self.modules["fork"] = Fork(self.rows_out(), self.cols_out(),
                    self.channels_in()//(self.coarse_in*self.coarse_group),
                    [self.fine, 1], self.coarse_out, backend=self.backend, regression_model=self.regression_model)

            if len(self.sparsity) == 0:
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
            else:
                self.modules["vector_dot"] = SparseVectorDot(self.rows_out(), self.cols_out(),
                    self.channels_in()//(self.coarse_in*self.coarse_group),
                    self.filters//(self.coarse_out*self.groups),
                    self.kernel_size, self.sparsity, self.window_sparsity, self.skipping_windows, self.fine,
                    backend=self.backend, regression_model=self.regression_model)

                self.modules["accum"] = Accum(self.rows_out(), self.cols_out(),
                        self.channels_in()//(self.coarse_in*self.coarse_group),
                        self.filters//(self.coarse_out*self.groups), 1, skipping_windows = self.skipping_windows,
                        window_sparsity = self.window_sparsity, backend=self.backend, regression_model=self.regression_model)

        self.modules["glue"] = Glue(self.rows_out(), self.cols_out(), 1,
                int(self.filters/self.coarse_out), self.coarse_in, self.coarse_out, self.coarse_group,
                backend=self.backend, regression_model=self.regression_model) # TODO

        self.modules["bias"] = Bias(self.rows_out(), self.cols_out(), 1, self.filters//(self.coarse_out*self.coarse_group),
                backend=self.backend, regression_model=self.regression_model) # TODO

        self.modules["shift_scale"] = ShiftScale(self.rows_out(), self.cols_out(), 1, self.filters//(self.coarse_out*self.coarse_group))

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

        if len(self.sparsity) > 0:
            # module sparsity depends on number of streams
            self.update()

    @coarse_group.setter
    def coarse_group(self, val: int) -> None:
        assert(val in self.get_coarse_group_feasible())
        self._coarse_group = val

        if len(self.sparsity) > 0:
            # module sparsity depends on number of streams
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

    def get_stream_sparsity(self, interleave=True):
        # Cycles taken for a window based on number of non-zeros and fine
        cycles_per_bin = np.ceil(np.flip(np.arange(self.kernel_size[0]*self.kernel_size[1] + 1))/self.fine) 

        # If you're not skipping all-zero windows, they still take one cycle
        if not (self.skipping_windows):
            cycles_per_bin[-1] = 1

        # Multiply the cycles per bin by the probability of each number of non-zeros, sum up the cycles and calculate the rate accordingly
        rate_per_channel = 1 / np.sum(cycles_per_bin*self.sparsity, axis = 1)

        #Balance the channels according to their rates
        if interleave:
            indices = np.argsort(rate_per_channel)
            indices = np.reshape(indices, (self.channels_in()//self.streams_in(), self.streams_in()))
            indices[1::2, :] = indices[1::2, ::-1] # reverse every other row
            indices = indices.flatten()
        else:
            indices = list(range(self.channels_in()))

        stream_sparsity = np.reshape([self.sparsity[i, :] for i in indices], (self.channels_in()//self.streams_in(), self.streams_in(), self.kernel_size[0]*self.kernel_size[1]+1)).mean(axis=0)
        stream_window_sparsity = np.reshape([self.window_sparsity[i] for i in indices], (self.channels_in()//self.streams_in(), self.streams_in())).mean(axis = 0)
        return stream_sparsity, stream_window_sparsity

    # def pipeline_depth(self):
    #     # pipeline depth of the sliding window minus the total words in the pipeline from padding
    #     # plus the words needed to fill the accum buffer
    #     return (self.kernel_rows-1)*(self.cols+self.pad_left+self.pad_right)*self.channels//self.coarse_in + \
    #             (self.kernel_cols-1)*self.channels//self.coarse_in - \
    #             ( self.pad_top * self.cols * self.channels//self.coarse_in + \
    #             (self.pad_left+self.pad_right)*self.channels//self.coarse_in ) + \
    #             self.channels//self.coarse_in

    def update(self):

        # sliding window
        self.modules['sliding_window'].rows     = self.rows
        self.modules['sliding_window'].cols     = self.cols
        self.modules['sliding_window'].channels = self.channels//(self.coarse_in*self.coarse_group)
        self.modules['sliding_window'].data_width   = self.input_t.width

        if self.backend == "chisel":
            # squeeze
            self.modules['squeeze'].rows     = self.rows_out()
            self.modules['squeeze'].cols     = self.cols_out()
            self.modules['squeeze'].channels = self.channels//(self.coarse_in*self.coarse_group)
            self.modules['squeeze'].coarse_out = self.fine
            self.modules['squeeze'].data_width = self.input_t.width

        # fork
        self.modules['fork'].rows     = self.rows_out()
        self.modules['fork'].cols     = self.cols_out()
        self.modules['fork'].channels = self.channels_in()//(self.coarse_in*self.coarse_group)
        self.modules['fork'].coarse   = self.coarse_out
        self.modules['fork'].data_width     = self.input_t.width
        if self.backend == "chisel":
            self.modules['fork'].kernel_size = [self.fine, 1]

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
            self.modules['vector_dot'].skipping_windows     = self.skipping_windows

            if len(self.sparsity) == 0:
                self.modules['vector_dot'].channels = (
                    self.channels*self.kernel_size[0]*self.kernel_size[1])//(
                    self.fine*self.coarse_in*self.coarse_group)
            else:
                self.modules['vector_dot'].channels = self.channels_in()//(self.coarse_in*self.coarse_group)
                self.modules['vector_dot'].sparsity = self.get_stream_sparsity()[0]
                self.modules['vector_dot'].window_sparsity = self.get_stream_sparsity()[1]

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
            if len(self.sparsity) == 0:
                self.modules['accum'].channels = (
                    self.channels*self.kernel_size[0]*self.kernel_size[1])//(
                    self.fine*self.coarse_in*self.coarse_group)
            else:
                self.modules['accum'].channels = self.channels//(self.coarse_in*self.coarse_group)
                self.modules['accum'].window_sparsity = self.get_stream_sparsity()[1]
                self.modules['accum'].skipping_windows = self.skipping_windows

        # glue
        self.modules['glue'].rows       = self.rows_out()
        self.modules['glue'].cols       = self.cols_out()
        self.modules['glue'].filters    = self.filters//self.coarse_group
        self.modules['glue'].coarse_in  = self.coarse_in
        self.modules['glue'].coarse_out = self.coarse_out
        self.modules['glue'].coarse_group = self.coarse_group
        self.modules['glue'].data_width = self.acc_t.width

        # bias
        self.modules['bias'].rows           = self.rows_out()
        self.modules['bias'].cols           = self.cols_out()
        self.modules['bias'].filters        = self.filters//(self.coarse_out*self.coarse_group)
        self.modules['bias'].data_width     = self.output_t.width
        self.modules['bias'].biases_width   = self.acc_t.width

        # shift scale
        self.modules['shift_scale'].rows           = self.rows_out()
        self.modules['shift_scale'].cols           = self.cols_out()
        self.modules['shift_scale'].filters        = self.filters//(self.coarse_out*self.coarse_group)
        self.modules['shift_scale'].data_width     = self.output_t.width
        self.modules['shift_scale'].biases_width   = self.acc_t.width

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
        if len(self.sparsity) == 0:
            parameters.fine  = self.fine
            parameters.sparsity.extend([])
        else:
            assert self.backend == "chisel"
            parameters.fine = self.modules["vector_dot"].fine
            if (self.skipping_windows):
                original_sparsity = np.hstack((self.sparsity[:, :-1], self.window_sparsity[:, np.newaxis]))
                parameters.sparsity.extend(original_sparsity.flatten())
            else:
                parameters.sparsity.extend(self.sparsity.flatten())
        parameters.use_uram     = self.use_uram
        parameters.block_floating_point    = self.block_floating_point
        parameters.skipping_windows = self.skipping_windows

        self.input_t.to_protobuf(parameters.input_t)
        self.output_t.to_protobuf(parameters.output_t)
        self.weight_t.to_protobuf(parameters.weight_t)
        self.acc_t.to_protobuf(parameters.acc_t)
        parameters.data_t.Clear()

    def get_coarse_group_feasible(self):
        return get_factors(self.groups)

    def get_coarse_in_feasible(self):
        return get_factors(int(self.channels_in())//self.groups)

    def get_coarse_out_feasible(self):
        return get_factors(int(self.channels_out())//self.groups)

    def get_fine_feasible(self):
        if self.backend == "chisel":
            if len(self.sparsity) == 0:
                return get_factors(self.kernel_size[0]*self.kernel_size[1])
            else:
                return list(range(1,self.kernel_size[0]*self.kernel_size[1]+1))
        elif self.backend == "hls":
            if self.kernel_size[0] != self.kernel_size[1]:
                # assert(self.kernel_size[0] == 1 or self.kernel_size[1] == 1)
                return [ 1, max(self.kernel_size[0],self.kernel_size[1])]
            else:
                return [ 1, self.kernel_size[0], self.kernel_size[0]*self.kernel_size[1] ]

    def get_weights_reloading_feasible(self):
        return get_factors(self.filters//(self.groups*self.coarse_out))

    def get_parameters_size(self):
        weights_size = self.channels_in() * ( self.filters // self.groups ) * self.kernel_size[0] * self.kernel_size[1]
        bias_size = 0
        return {
            "weights"   : weights_size,
            "bias"      : bias_size
        }

    def get_operations(self):
        return self.kernel_size[0]*self.kernel_size[1]*self.channels_in()*self.filters*self.rows_out()*self.cols_out()

    def resource(self):

        if self.backend == "chisel":

            # get module resource models
            sw_rsc          = self.modules['sliding_window'].rsc()
            squeeze_rsc     = self.modules['squeeze'].rsc()
            fork_rsc        = self.modules['fork'].rsc()
            vector_dot_rsc  = self.modules['vector_dot'].rsc()
            accum_rsc       = self.modules['accum'].rsc()
            glue_rsc        = self.modules['glue'].rsc()
            bias_rsc        = self.modules['bias'].rsc()
            shift_scale_rsc = self.modules['shift_scale'].rsc()

            # remove redundant modules
            if self.kernel_size[0] == 1 and self.kernel_size[1] == 1:
                sw_rsc      = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            if self.fine == self.kernel_size[0]*self.kernel_size[1] or len(self.sparsity) > 0:
                # when sparsity occurs, the crossbar in sparse_vector_dot already acts as a squeeze
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
            if self.weight_t.width <= 8 and self.input_t.width <= 8:
                vector_dot_rsc["DSP"] = vector_dot_rsc["DSP"]*0.5
            elif self.weight_t.width <= 4 and self.input_t.width <= 4:
                vector_dot_rsc["DSP"] = vector_dot_rsc["DSP"]*0.25

            # accumulate resource usage based on coarse factors
            rsc = { rsc_type: (
                sw_rsc[rsc_type]*self.coarse_in*self.coarse_group +
                squeeze_rsc[rsc_type]*self.coarse_in*self.coarse_group +
                fork_rsc[rsc_type]*self.coarse_in*self.coarse_group +
                math.ceil(vector_dot_rsc[rsc_type]*self.coarse_in*self.coarse_out*self.coarse_group) +
                accum_rsc[rsc_type]*self.coarse_in*self.coarse_out*self.coarse_group +
                glue_rsc[rsc_type] +
                bias_rsc[rsc_type]*self.coarse_out*self.coarse_group +
                shift_scale_rsc[rsc_type]*self.coarse_out*self.coarse_group
            ) for rsc_type in ["LUT", "FF", "DSP", "BRAM"] }

        # weight usage
        weight_memory_depth = float((self.filters/self.groups)* \
                                    self.channels_in()* \
                                    self.kernel_size[0]* \
                                    self.kernel_size[1]) / \
            float(self.fine*self.coarse_in*self.coarse_out*self.coarse_group)

        if self.double_buffered:
            weight_memory_depth *= 2

        if self.use_uram:
            array_resource_model = bram_array_resource_model
        else:
            array_resource_model = uram_array_resource_model

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

        if self.stream_weights:
            weights_bram_usage = 0
        elif self.use_uram:
            weights_uram_usage = uram_array_resource_model(weight_array_depth, weight_array_width) * weight_array_num
            rsc["URAM"] = weights_uram_usage
            weights_bram_usage = 0
        else:
            weights_bram_usage = bram_array_resource_model(weight_array_depth, weight_array_width, "memory") * weight_array_num

        # bias usage
        if self.has_bias:
            bias_memory_depth = float(self.filters) / float(self.coarse_out*self.coarse_group)
            biases_bram_usage = bram_array_resource_model(
                        int(bias_memory_depth),self.acc_t.width,
                        "memory") * self.coarse_out * self.coarse_group
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

        # return total resource
        return rsc

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

        assert bias.shape[0] == self.filters  ,     "ERROR (bias): invalid filter dimension"

        # instantiate convolution layer
        convolution_layer = torch.nn.Conv2d(self.channels_in(), self.filters, self.kernel_size,
                stride=self.stride, padding=0, groups=self.groups)

        # update weights
        convolution_layer.weight = torch.nn.Parameter(torch.from_numpy(weights))

        # update bias
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
        data = convolution_layer(data).detach().numpy()
        print(data.shape)
        return data
        # return convolution_layer(data).detach().numpy()

