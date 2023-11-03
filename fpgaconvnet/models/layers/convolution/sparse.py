import importlib
import math
from typing import Union, List
from dataclasses import dataclass, field

import pydot
import numpy as np
from dacite import from_dict

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

from fpgaconvnet.models.layers import Layer, Layer3D
from fpgaconvnet.models.modules import SparseVectorDot, Accum

from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY, SPARSITY


@dataclass(kw_only=True)
class ConvolutionLayerSparseBase:

    def get_squeeze_parameters(self):
        param = super().get_squeeze_parameters()
        param["coarse_in"] = np.prod(self.kernel_size)
        param["coarse_out"] = np.prod(self.kernel_size)
        return param

    def get_fine_feasible(self):
        return list(range(1, np.prod(self.kernel_size) + 1))

    def get_sparse_operations(self):
        return self.get_operations()*np.average(self.sparsity)


@dataclass(kw_only=True)
class ConvolutionLayerTraitSparse(ConvolutionLayerSparseBase):
    sparsity: list = field(default_factory=lambda: [], init=True)
    skip_zero_windows: bool = False

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        # check that sparsity has the right number of elements
        assert len(np.flatten(self.sparsity)) == self.channels*(np.prod(self.kernel_size)+1), \
                "Sparsity must have channel number of elements"

        # reshape the sparsity
        self.sparsity = np.reshape(np.flatten(self.sparsity),
                (self.channels, (np.prod(self.kernel_size)+1)))

        # change the vector dot to a sparse equivalent
        self.modules["vector_dot"] = SparseVectorDot(self.rows_out(),
            self.cols_out(), self.channels_in()//self.streams_in(),
            self.filters//(self.coarse_out*self.groups), self.kernel_size,
            self.sparsity, self.skip_all_zero_window, self.fine,
            backend="chisel", regression_model=self.regression_model)

        # modify the accum module also
        self.modules["accum"] = Accum(self.rows_out(), self.cols_out(),
            self.channels_in()//(self.coarse_in*self.coarse_group),
            self.filters//(self.coarse_out*self.groups), 1,
            self.skip_all_zero_window, self.channel_sparsity_hist,
            backend="chisel", regression_model=self.regression_model)

        # update all the modules
        self.update_modules()

    def get_vector_dot_parameters(self):
        param = super().get_vector_dot_parameters()
        param["channels"] = self.channels_in()//self.streams_in()
        param["kernel_size"] = self.kernel_size
        param["sparsity_hist"] = self.sparsity
        return param

    def get_interleaving(self, method="opt"):

        match method:
            case "opt":

                # get the cycles per bin
                cycles_per_bin = np.ceil(np.flip(np.arange(np.prod(self.kernel_size) + 1))/self.fine)
                if not (self.skip_all_zero_window):
                    cycles_per_bin[-1] = 1

                # Multiply the cycles per bin by the probability of each number of non-zeros,
                # sum up the cycles and calculate the rate accordingly
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
                    np.prod(self.kernel_size)+1)).mean(axis=0)

        # return the sparsity and interleaving
        return stream_sparsity, indices

    def latency(self, metric="avg", interleaving="opt"):

        # get the total workload in terms of vector-dot products
        workload = np.prod(self.shape_out[:-1])* \
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
                    zero_window_cycles = ws[i,0]
                    if self.skip_all_zero_window:
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

        # return the slowest of operation latency and data movement latency
        return int(math.ceil(max([
            operation_latency,
            self.rows_in()*self.rows_in()*self.channels_in()//self.streams_in(),
            self.rows_out()*self.rows_out()*self.channels_out()//self.streams_out(),
        ])))


@dataclass(kw_only=True)
class ConvolutionLayerTraitSparsePointwise(ConvolutionLayerSparseBase):
    channel_sparsity: list = field(default_factory=lambda: [], init=True)
    clusters: int = 1

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        # check that the kernel size is 1
        assert np.prod(self.kernel_size) == 1, \
                "Pointwise sparse layers must have a kernel size of 1"

        # check that the channel sparsity has channel number of elements
        assert len(self.channel_sparsity) == self.channels, \
                "Channel sparsity must have channel number of elements"

    def get_interleaving(self, method="opt"):

        match method:
            case "opt":

                #Balance the channels according to their sparsity
                indices = np.argsort(self.channel_sparsity_avg)
                indices = np.reshape(indices,
                        (self.channels_in()//self.streams_in(), self.streams_in()))
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


