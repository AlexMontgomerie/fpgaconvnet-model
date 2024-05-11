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
            channel_sparsity_avg: list = [],
            clusters: int = 1,
            block_floating_point: bool = False,
            backend: str = "chisel", # default to no bias for old configs
            regression_model: str = "linear_regression",
            stream_weights: int = 0,
            use_uram: bool = False,
            input_compression_ratio: list = [1.0],
            output_compression_ratio: list = [1.0],
            weight_compression_ratio: list = [1.0]
        ):

        # reshape into window sparsity per channel
        self.channel_sparsity_avg = channel_sparsity_avg

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
            regression_model=regression_model,
            stream_weights=stream_weights,
            use_uram=use_uram,
            input_compression_ratio=input_compression_ratio,
            output_compression_ratio=output_compression_ratio,
            weight_compression_ratio=weight_compression_ratio
        )

        # data packing not supported for sparse layers
        self.data_packing = False

        # update modules
        self.update()


    def get_interleaving(self, method="opt"):

        match method:
            case "opt":

                #Balance the channels according to their sparsity
                indices = np.argsort(self.channel_sparsity_avg)
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
        stream_sparsity = np.reshape([self.channel_sparsity_avg[i] for i in indices],
                (self.channels_in()//self.streams_in(), self.streams_in())).mean(axis=0)

        # return the sparsity and interleaving
        return stream_sparsity, indices

    def get_average_cluster_sparsity(self, interleaving="naive", clustering="naive"):

        # get the interleaving and grouping
        indices = self.get_interleaving(method=interleaving)
        clusters = self.get_clustering(method=clustering)

        # reshape the sparsity into the interleaved streams
        stream_sparsity = np.reshape([self.channel_sparsity_avg[i] for i in indices],
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
                # FIXME: this is a workaround
                # todo: make dsp usage a separate variable <= coarse_in*coarse_group*courses_out
                cycles = np.subtract(1, cluster_sparsity) 

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
        return self.get_operations()*np.average(self.channel_sparsity_avg)

    def layer_info(self,parameters,batch_size=1):
        super().layer_info(parameters, batch_size)
        parameters.sparsity.extend(self.channel_sparsity_avg)
        parameters.clusters = self.clusters

    def resource(self):
        # get module resource models
        rsc = super().resource()
        # when sparsity occurs, the crossbar in sparse_vector_dot already acts as a squeeze
        squeeze_rsc = self.modules['squeeze'].rsc()
        for rsc_type in squeeze_rsc.keys():
            if rsc_type in rsc.keys():
                rsc[rsc_type] -= squeeze_rsc[rsc_type]
        return rsc
