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
            channel_sparsity_hist: list = [],
            skip_all_zero_window: bool = True,
            block_floating_point: bool = False,
            backend: str = "chisel", # default to no bias for old configs
            regression_model: str = "linear_regression",
            stream_weights: int = 0,
            use_uram: bool = False,
            input_compression_ratio: list = [1.0],
            output_compression_ratio: list = [1.0],
            weight_compression_ratio: list = [1.0]
        ):

        # flag indicating zero window skipping used
        self.skip_all_zero_window = skip_all_zero_window
        #Ensure histogram data provided
        channel_sparsity_hist = np.reshape(channel_sparsity_hist, (channels, int(kernel_rows*kernel_cols+1))) 
        self.channel_sparsity_hist = channel_sparsity_hist

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
            regression_model=regression_model,
            stream_weights=stream_weights,
            use_uram=use_uram,
            input_compression_ratio=input_compression_ratio,
            output_compression_ratio=output_compression_ratio,
            weight_compression_ratio=weight_compression_ratio
        )

        # data packing not supported for sparse layers
        self.data_packing = False

        # change modules to sparse equivalents
        self.modules["vector_dot"] = SparseVectorDot(self.rows_out(), self.cols_out(),
            self.channels_in()//(self.coarse_in*self.coarse_group),
            self.filters//(self.coarse_out*self.groups),
            self.kernel_size, self.channel_sparsity_hist, self.skip_all_zero_window, self.fine,
            backend=self.backend, regression_model=self.regression_model)
        self.modules["accum"] = Accum(self.rows_out(), self.cols_out(),
            self.channels_in()//(self.coarse_in*self.coarse_group),
            self.filters//(self.coarse_out*self.groups), 1, 
            self.skip_all_zero_window, self.channel_sparsity_hist, 
            backend=self.backend, regression_model=self.regression_model)

        # update modules
        self.update()


    def get_interleaving(self, method="opt"):

        match method:
            case "opt":

                # get the cycles per bin
                cycles_per_bin = np.ceil(np.flip(np.arange(self.kernel_size[0]*self.kernel_size[1] + 1))/self.fine)
                if not (self.skip_all_zero_window):
                    cycles_per_bin[-1] = 1

                # Multiply the cycles per bin by the probability of each number of non-zeros, sum up the cycles and calculate the rate accordingly
                rate_per_channel = 1 / np.sum(cycles_per_bin*self.channel_sparsity_hist, axis = 1)

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
        stream_sparsity_hist = np.reshape([self.channel_sparsity_hist[i, :] for i in indices],
                (self.channels_in()//self.streams_in(), self.streams_in(),
                    self.kernel_size[0]*self.kernel_size[1]+1)).mean(axis=0)

        # return the sparsity and interleaving
        return stream_sparsity_hist, indices

    def latency(self, metric="avg", interleaving="opt"):

        # get the total workload in terms of vector-dot products
        workload = self.rows_out()*self.cols_out()* \
                (self.channels_in()/(self.coarse_in*self.coarse_group))* \
                (self.filters/(self.coarse_out*self.groups))

        # get the latency from performing the operations
        match metric:
            case "avg":

                # get the stream sparsity
                ws, _ = self.get_stream_sparsity(interleaving=interleaving)

                # operation latency for each stream
                operation_latency_per_stream = []

                # iterate over streams
                for i in range(ws.shape[0]):

                    # average cycles spent on complete zero windows
                    zero_window_cycles = ws[i,-1]
                    if self.skip_all_zero_window:
                        zero_window_cycles = 0

                    # get the average number of cycles for the each vector dot product
                    vector_dot_cycles = \
                            zero_window_cycles + \
                            sum([ math.ceil((np.prod(self.kernel_size)-j)/self.fine)*ws[i,j]
                                for j in range(0, np.prod(self.kernel_size)) ])

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
        self.modules['vector_dot'].sparsity_hist = self.get_stream_sparsity()[0]

        # accum
        self.modules['accum'].channels = self.channels//self.streams_in()
        self.modules['accum'].sparsity_hist = self.get_stream_sparsity()[0]

    def get_fine_feasible(self):
        return list(range(1,self.kernel_size[0]*self.kernel_size[1]+1))

    def get_sparse_operations(self):
        return self.get_operations()*np.average(self.channel_sparsity_hist)

    def layer_info(self,parameters,batch_size=1):
        super().layer_info(parameters, batch_size)
        parameters.sparsity.extend(self.channel_sparsity_hist.flatten())
        parameters.skip_all_zero_window = self.skip_all_zero_window

    def resource(self):
        # get module resource models
        rsc = super().resource()
        # when sparsity occurs, the crossbar in sparse_vector_dot already acts as a squeeze
        squeeze_rsc = self.modules['squeeze'].rsc()
        for rsc_type in squeeze_rsc.keys():
            if rsc_type in rsc.keys():
                rsc[rsc_type] -= squeeze_rsc[rsc_type]
        return rsc

