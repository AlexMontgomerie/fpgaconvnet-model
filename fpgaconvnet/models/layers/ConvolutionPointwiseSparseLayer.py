import importlib
import math
from typing import Union, List

import pydot
import numpy as np

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model
from fpgaconvnet.models.layers.ConvolutionLayer import ConvolutionLayer

from fpgaconvnet.models.modules import SparseVectorDot
from fpgaconvnet.models.modules import Accum

class ConvolutionPointwiseSparseLayer(ConvolutionLayer):

    def __init__(
            self,
            filters: int,
            rows: int,
            cols: int,
            channels: int,
            coarse_in: int = 1,
            coarse_out: int = 1,
            coarse_group: int = 1,
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
            clusters: int = 1,
            block_floating_point: bool = False,
            backend: str = "chisel", # default to no bias for old configs
            regression_model: str = "linear_regression"
        ):

        # reshape into window sparsity per channel
        self.sparsity = np.array(sparsity).reshape(channels)

        # save the number of clusters
        self.clusters = clusters

        # initialise convolution class
        super().__init__(
            filters,rows, cols, channels,
            coarse_in=coarse_in,
            coarse_out=coarse_out,
            coarse_group=coarse_group,
            kernel_rows=1,
            kernel_cols=1,
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

        # update modules
        self.update()


    def get_interleaving(self, method="opt"):

        match method:
            case "opt":

                #Balance the channels according to their sparsity
                indices = np.argsort(self.sparsity)
                indices = np.reshape(indices, (self.channels_in()//self.streams_in(), self.streams_in()))
                indices[1::2, :] = indices[1::2, ::-1] # reverse every other row
                indices = indices.flatten()

                # return the indices
                return indices

            case "naive":
                return np.arange(self.channels_in())
            case _:
                raise ValueError(f"method {method} not supported!")

    def get_clustering(self, method="naive"):

        match method:
            case "naive":

                # get an index for each stream in
                indices = np.arange(self.coarse_in*self.coarse_group)

                # return the reshaped indices
                return np.reshape(indices, (self.clusters, -1))

            case _:
                raise ValueError(f"grouping method {method} not supported!")


    def get_stream_sparsity(self, interleaving="opt"):

        # get the interleaving
        indices = self.get_interleaving(method=interleaving)

        # get the stream sparsity
        stream_sparsity = np.reshape([self.sparsity[i] for i in indices],
                (self.channels_in()//self.streams_in(), self.streams_in())).mean(axis=0)

        # return the sparsity and interleaving
        return stream_sparsity, indices

    def get_average_cluster_sparsity(self, interleaving="naive", clustering="naive"):

        # get the interleaving and grouping
        indices = self.get_interleaving(method=interleaving)
        clusters = self.get_clustering(method=clustering)

        # reshape the sparsity into the interleaved streams
        stream_sparsity = np.reshape([self.sparsity[i] for i in indices],
                (self.channels_in()//self.streams_in(), self.streams_in()))

        # reshape into cluster sparsity
        cluster_sparsity = [ np.mean([stream_sparsity[:,i] for i in cluster ]) \
                for cluster in clusters ]

        # return the cluster sparsity
        return cluster_sparsity

    def latency(self, metric="avg", interleaving="naive", clustering="naive"):

        # get the total workload in terms of vector-dot products
        workload = self.rows_out()*self.cols_out()* \
                (self.channels_in()/self.streams_in())* \
                (self.filters/(self.coarse_out*self.groups))

        # get the number of streams per cluster
        cluster_streams = self.streams_in()//self.clusters

        # get the latency from performing the operations
        match metric:
            case "avg":

                # get the average sparsity per cluster
                cluster_sparsity = self.get_average_cluster_sparsity(
                        interleaving=interleaving, clustering=clustering)

                # set a ceiling on the sparsity
                cluster_sparsity = np.minimum(cluster_sparsity, 1-(1/cluster_streams))

                # calculate average cycles per cluster
                cycles = np.multiply(np.subtract(1, cluster_sparsity),
                        float(cluster_streams/self.fine))

                # get the max latency for each stream
                operation_latency = workload * max(cycles)

            case "min":
                operation_latency = workload
            case "max":
                operation_latency = workload*cluster_streams
            case _:
                raise ValueError(f"metric {metric} not supported!")

        # return the slowest of operation latency and data movement latency
        return math.ceil(max([
            operation_latency,
            self.rows_in()*self.rows_in()*self.channels_in()/self.streams_in(),
            self.rows_out()*self.rows_out()*self.channels_out()/self.streams_out(),
        ]))

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

