import importlib
import math
from typing import Union, List

import pydot
import numpy as np

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model
from fpgaconvnet.models.layers import ConvolutionLayer

from fpgaconvnet.models.modules import SparseVectorDot
from fpgaconvnet.models.modules import Accum

class ConvolutionSparseLayer(ConvolutionLayer):

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

        # flag indicating zero window skipping used
        self.skipping_windows = skipping_windows

        self.window_sparsity = []
        self.sparsity = sparsity

        #Ensure histogram data provided
        assert (len(sparsity) == channels*(kernel_cols*kernel_rows+1))

        # reshape into window sparsity per channel
        self.sparsity = np.array(sparsity).reshape((channels, kernel_rows*kernel_cols+1))
        self.window_sparsity = np.copy(np.squeeze(self.sparsity[:, -1]))


        # initialise convolution class
        super().__init__(
            filters,rows, cols, channels,
            coarse_in=coarse_in,
            coarse_out=coarse_out,
            coarse_group=coarse_group,
            kernel_rows=kernel_rows,
            kernel_cols=kernel_cols,
            stride_rows=stride_rows,
            stride_cols=stride_cols,
            groups=groups,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
            pad_left=pad_left,
            pad_right=pad_right,
            fine=fine,
            input_t=input_t,
            output_t=output_t,
            weight_t=weight_t,
            acc_t=acc_t,
            has_bias=has_bias,
            block_floating_point=block_floating_point,
            backend=backend,
            regression_model=regression_model
        )

        # change modules to sparse equivalents
        self.modules["vector_dot"] = SparseVectorDot(self.rows_out(), self.cols_out(),
            self.channels_in()//(self.coarse_in*self.coarse_group),
            self.filters//(self.coarse_out*self.groups),
            self.kernel_size, self.sparsity, self.window_sparsity, self.skipping_windows, self.fine,
            backend=self.backend, regression_model=self.regression_model)
        self.modules["accum"] = Accum(self.rows_out(), self.cols_out(),
                self.channels_in()//(self.coarse_in*self.coarse_group),
                self.filters//(self.coarse_out*self.groups), 1, skipping_windows = self.skipping_windows,
                window_sparsity = self.window_sparsity, backend=self.backend, regression_model=self.regression_model)

        # update modules
        self.update()


    def get_interleaving(self, method="opt"):

        match method:
            case "opt":

                # get the cycles per bin
                cycles_per_bin = np.ceil(np.flip(np.arange(self.kernel_size[0]*self.kernel_size[1] + 1))/self.fine)
                if not (self.skipping_windows):
                    cycles_per_bin[-1] = 1

                # Multiply the cycles per bin by the probability of each number of non-zeros, sum up the cycles and calculate the rate accordingly
                rate_per_channel = 1 / np.sum(cycles_per_bin*self.sparsity, axis = 1)

                #Balance the channels according to their rates
                indices = np.argsort(rate_per_channel)
                indices = np.reshape(indices, (self.channels_in()//self.streams_in(), self.streams_in()))
                indices[1::2, :] = indices[1::2, ::-1] # reverse every other row
                indices = indices.flatten()

                # return the indices
                return indices

            case "naive":
                return list(range(self.channels_in()))
            case _:
                raise ValueError(f"method {method} not supported!")

    def get_stream_sparsity(self, interleaving="opt"):

        # get the interleaving
        indices = self.get_interleaving(method=interleaving)

        # get the stream sparsity
        stream_sparsity = np.reshape([self.sparsity[i, :] for i in indices],
                (self.channels_in()//self.streams_in(), self.streams_in(),
                    self.kernel_size[0]*self.kernel_size[1]+1)).mean(axis=0)

        # get stream window sparsity
        stream_window_sparsity = np.reshape([self.window_sparsity[i] for i in indices],
                (self.channels_in()//self.streams_in(), self.streams_in())).mean(axis = 0)

        # return the sparsity and interleaving
        return stream_sparsity, stream_window_sparsity, indices

    def latency(self, metric="avg", interleaving="opt"):

        # get the total workload in terms of vector-dot products
        workload = self.rows_out()*self.cols_out()* \
                (self.channels_in()/(self.coarse_in*self.coarse_group))* \
                (self.filters/(self.coarse_out*self.groups))

        # get the latency from performing the operations
        match metric:
            case "avg":

                # get the stream sparsity
                ws, _, _ = self.get_stream_sparsity(interleaving=interleaving)

                # operation latency for each stream
                operation_latency_per_stream = []

                # iterate over streams
                for i in range(ws.shape[0]):

                    # average cycles spent on complete zero windows
                    zero_window_cycles = ws[i,0]
                    if self.skipping_windows:
                        zero_window_cycles = 0

                    # get the average number of cycles for the each vector dot product
                    vector_dot_cycles = \
                            zero_window_cycles + \
                            sum([ math.ceil(j/self.fine)*ws[i,j]
                                for j in range(1, np.prod(self.kernel_size)+1) ])

                    # append the operation latency for the stream
                    operation_latency_per_stream.append(
                            workload*vector_dot_cycles)

                # get the max operation latency
                operation_latency = max(operation_latency_per_stream)

            case "min":
                operation_latency = workload
            case "max":
                operation_latency = workload*np.prod(self.kernel_size)
            case _:
                raise ValueError(f"metric {metric} not supported!")

        # sliding window rate slow down
        slwin_rate = self.rows_in()*self.cols_in()/(self.rows_out()*self.cols_out())

        # return the slowest of operation latency and data movement latency
        return math.ceil(max([
            # slwin_rate*operation_latency,
            operation_latency,
            self.rows_in()*self.rows_in()*self.channels_in()/self.streams_in(),
            self.rows_out()*self.rows_out()*self.channels_out()/self.streams_out(),
        ]))

    def update(self):

        # update the base convolution modules
        super().update()

        # vector dot
        self.modules['vector_dot'].channels = self.channels_in()//self.streams_in()
        self.modules['vector_dot'].sparsity = self.get_stream_sparsity()[0]
        self.modules['vector_dot'].window_sparsity = self.get_stream_sparsity()[1]

        # accum
        self.modules['accum'].channels = self.channels//self.streams_in()
        self.modules['accum'].window_sparsity = self.get_stream_sparsity()[1]
        self.modules['accum'].skipping_windows = self.skipping_windows

    def get_fine_feasible(self):
        return list(range(1,self.kernel_size[0]*self.kernel_size[1]+1))

    def get_sparse_operations(self):
        return self.get_operations()*np.average(self.sparsity)

    def resource(self):

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
            vector_dot_rsc[rsc_type]*self.coarse_in*self.coarse_out*self.coarse_group +
            accum_rsc[rsc_type]*self.coarse_in*self.coarse_out*self.coarse_group +
            glue_rsc[rsc_type] +
            bias_rsc[rsc_type]*self.coarse_out*self.coarse_group +
            shift_scale_rsc[rsc_type]*self.coarse_out*self.coarse_group
        ) for rsc_type in ["LUT", "FF", "DSP", "BRAM"] }

        # weight usage
        weight_memory_depth = math.ceil(
            float((self.filters/self.groups)* \
                self.channels_in()* \
                self.kernel_size[0]* \
                self.kernel_size[1]) / \
                    float(self.fine*self.coarse_in*\
                    self.coarse_out*self.coarse_group))

        if self.double_buffered:
            weight_memory_depth *= 2

        if self.use_uram:
            array_resource_model = bram_array_resource_model
        else:
            array_resource_model = uram_array_resource_model

        if self.data_packing:
            weight_array_depth = math.ceil(weight_memory_depth)
            weight_array_width = self.weight_t.width*self.fine*self.coarse_out
            weight_array_num = self.coarse_in*self.coarse_group
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

