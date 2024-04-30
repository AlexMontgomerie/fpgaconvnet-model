from typing import ClassVar
from dataclasses import dataclass
from collections import OrderedDict
import math

import pydot
import numpy as np
from dacite import from_dict
import networkx as nx

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model
from fpgaconvnet.models.layers.convolution.base import ConvolutionLayer2DMixin
from fpgaconvnet.models.layers.convolution.backend import ConvolutionLayerChiselMixin
from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY

@dataclass(kw_only=True)
class ConvolutionLayerSparseChisel(ConvolutionLayerChiselMixin, ConvolutionLayer2DMixin):

    sparsity: list[float]
    interleaving_method: str = "opt"
    latency_metric: str = "avg"

    name: ClassVar[str] = "convolution_sparse"
    backend: ClassVar[BACKEND] = BACKEND.CHISEL
    register: ClassVar[bool] = True

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        # check that sparsity has the right number of elements
        assert len(np.flatten(self.sparsity)) == self.channels*(math.prod(self.kernel_size)+1), \
                    "Sparsity must have channel number of elements"

        # reshape the sparsity
        self.sparsity = np.reshape(np.flatten(self.sparsity),
               (self.channels, (math.prod(self.kernel_size)+1)))

    def get_fine_feasible(self):
        return list(range(1, math.prod(self.kernel_size) + 1))

    def get_stream_sparsity(self, indices):

        # get the stream sparsity
        stream_sparsity = np.reshape([self.sparsity[i, :] for i in indices],
                (self.channels_in()//self.streams_in(), self.streams_in(),
                    math.prod(self.kernel_size)+1)).mean(axis=0)

        # return the sparsity and interleaving
        return stream_sparsity, indices

    def latency(self, indices=None):

        # get the total workload in terms of vector-dot products
        workload = math.prod(self.output_shape()[:-1])* \
                (self.channels_in()/(self.streams_in()))* \
                (self.filters/(self.coarse_out*self.groups))

        # get the latency from performing the operations
        match self.latency_metric:
            case "avg":

                if indices == None:
                    # get the interleaving indices
                    indices = get_interleaving(self, method=self.interleaving_method)

                # get the stream sparsity
                ws, _ = self.get_stream_sparsity(indices)

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
                                for j in range(1, math.prod(self.kernel_size)+1) ])

                    # append the operation latency for the stream
                    operation_latency_per_stream.append(
                            workload*vector_dot_cycles)

                # get the max operation latency
                operation_latency = max(operation_latency_per_stream)

            case "min":
                operation_latency = workload

            case "max":
                operation_latency = workload*math.prod(self.kernel_size)

            case _:
                raise ValueError(f"metric {self.latency_metric} not supported!")

        # return the slowest of operation latency and data movement latency
        return int(math.ceil(max([
                operation_latency,
                math.prod(self.input_shape())//self.streams_in(),
                math.prod(self.output_shape())//self.streams_out(),
            ])))

    @property
    def module_lookup(self) -> OrderedDict:
        return OrderedDict({
            "pad": self.get_pad_parameters,
            "sliding_window": self.get_sliding_window_parameters,
            "fork": self.get_fork_parameters,
            "sparse_vector_dot": self.get_sparse_vector_dot_parameters,
            "accum": self.get_accum_parameters,
            "glue": self.get_glue_parameters,
            "bias": self.get_bias_parameters
        })

    def get_pad_parameters(self):
        param = super().get_pad_parameters()
        param["streams"] = 1
        return param

    def get_sliding_window_parameters(self):
        param = super().get_sliding_window_parameters()
        param["streams"] = 1
        return param

    def get_fork_parameters(self):
        return {
            "repetitions": math.prod(self.output_shape()[:-1])*self.channels//self.streams_in(),
            "streams": 1,
            "fine": math.prod(self.kernel_size),
            "coarse": self.coarse_out,
            "data_t": self.input_t,
        }

    def get_sparse_vector_dot_parameters(self):

        return {
            "repetitions": math.prod(self.output_shape()[:-1]) * \
                self.channels//self.streams_in(),
            "streams": self.coarse_in*self.coarse_out*self.coarse_group,
            "filters": self.filters//self.streams_out(),
            "kernel_size": self.kernel_size,
            "sparsity": self.sparsity,
            "data_t": self.input_t,
            "weight_t": self.weight_t,
            "acc_t": self.acc_t,
        }

    def get_accum_parameters(self):

        channels = self.channels//self.streams_in() * \
                math.prod(self.kernel_size)//self.fine

        return {
            "repetitions": math.prod(self.output_shape()[:-1]),
            "streams": self.coarse_in*self.coarse_out*self.coarse_group,
            "channels": self.channels//self.streams_in(),
            "filters": self.filters//self.streams_out(),
            "data_t": self.acc_t,
        }

    def get_glue_parameters(self):
        param = super().get_glue_parameters()
        param["streams"] = 1
        return param

    def get_bias_parameters(self):
        param = super().get_bias_parameters()
        param["streams"] = 1
        return param

    def build_module_graph(self) -> nx.DiGraph:

        # get the module graph
        self.module_graph = nx.DiGraph()

        # add the modules
        for i in range(self.streams_in()):
            self.module_graph.add_node("pad_{i}", module=self.modules["pad"])
            self.module_graph.add_node("sliding_window_{i}", module=self.modules["sliding_window"])
            self.module_graph.add_node("fork_{i}", module=self.modules["fork"])
            self.module_graph.add_node("sparse_vector_dot_{i}", module=self.modules["sparse_vector_dot"])
            self.module_graph.add_node("accum_{i}", module=self.modules["accum"])
        for i in range(self.streams_out()):
            self.module_graph.add_node("glue_{i}", module=self.modules["glue"])
            self.module_graph.add_node("bias_{i}", module=self.modules["bias"])

        # connect the modules
        for i in range(self.streams_in()):
            self.module_graph.add_edge(f"pad_{i}", f"sliding_window_{i}")
            self.module_graph.add_edge(f"sliding_window_{i}", f"fork_{i}")
            self.module_graph.add_edge(f"fork_{i}", f"sparse_vector_dot_{i}")
            self.module_graph.add_edge(f"sparse_vector_dot_{i}", f"accum_{i}")
        # TODO: accum to glue connection
        for i in range(self.streams_out()):
            self.module_graph.add_edge(f"glue_{i}", f"bias_{i}")

@dataclass(kw_only=True)
class ConvolutionLayerSparseSkippingChisel(ConvolutionLayerSparseChisel):

    sparsity: list[float]

    name: ClassVar[str] = "convolution_sparse_skipping"
    backend: ClassVar[BACKEND] = BACKEND.CHISEL
    register: ClassVar[bool] = True

    @property
    def module_lookup(self) -> OrderedDict:
        return OrderedDict({
            "pad": self.get_pad_parameters,
            "sliding_window": self.get_sliding_window_parameters,
            "sparse_queue": self.get_sparse_queue_parameters,
            "fork": self.get_fork_parameters,
            "sparse_vector_dot": self.get_sparse_vector_dot_parameters,
            "sparse_accum": self.get_sparse_accum_parameters,
            "glue": self.get_glue_parameters,
            "bias": self.get_bias_parameters
        })

@dataclass(kw_only=True)
class ConvolutionLayerSparsePointwiseChisel(ConvolutionLayerChiselMixin, ConvolutionLayer2DMixin):

    kernel_rows: int = 1
    kernel_cols: int = 1
    channel_sparsity: list[float]
    clusters: int = 1
    interleaving_method: str = "opt"
    clustering_method: str = "opt"
    latency_metric: str = "avg"

    name: ClassVar[str] = "convolution_sparse_pointwise"
    backend: ClassVar[BACKEND] = BACKEND.CHISEL
    register: ClassVar[bool] = True

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        # check that the kernel size is 1
        assert math.prod(self.kernel_size) == 1, \
                "Pointwise sparse layers must have a kernel size of 1"

        # check that the channel sparsity has channel number of elements
        assert len(self.channel_sparsity) == self.channels, \
                "Channel sparsity must have channel number of elements"

    def get_average_cluster_sparsity(self, indices, clusters):

        # reshape the sparsity into the interleaved streams
        stream_sparsity = np.reshape([self.channel_sparsity[i] for i in indices],
                (self.channels_in()//self.streams_in(), self.streams_in()))

        # reshape into cluster sparsity
        cluster_sparsity = [ np.mean([stream_sparsity[:,i] for i in cluster ]) for cluster in clusters ]

        # return the cluster sparsity
        return cluster_sparsity

    def latency(self, indices = None, clusters = None):

        # get the total workload in terms of vector-dot products
        workload = math.prod(self.output_shape()[:-1])* \
                (self.channels_in()/(self.streams_in()))* \
                (self.filters/(self.coarse_out*self.groups))

        # get the number of streams per cluster
        cluster_streams = self.streams_in()//self.clusters

        # get the latency from performing the operations
        match self.latency_metric:
            case "avg":

                if indices == None:
                    # get the interleaving indices
                    indices = get_interleaving(self, method=self.interleaving_method)

                if clusters == None:
                    # get the clustering indices
                    clusters = get_clustering(self, method=self.clustering_method)

                # get the average sparsity per cluster
                cluster_sparsity = self.get_average_cluster_sparsity(indices, clusters)

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
                raise ValueError(f"metric {self.latency_metric} not supported!")

        # return the slowest of operation latency and data movement latency
        return int(math.ceil(max([
                operation_latency,
                math.prod(self.input_shape())//self.streams_in(),
                math.prod(self.output_shape())//self.streams_out(),
            ])))

    @property
    def module_lookup(self) -> OrderedDict:
        return OrderedDict({
            "pad": self.get_pad_parameters,
            # "stride": self.get_stride_parameters,
            "fork": self.get_fork_parameters,
            # "repeat": self.get_repeat_parameters,
            "sparse_vector_multiply": self.get_sparse_vector_multiply_parameters,
            "accum": self.get_accum_parameters,
            "glue": self.get_glue_parameters,
            "bias": self.get_bias_parameters
        })

    def get_pad_parameters(self):
        param = super().get_pad_parameters()
        param["streams"] = 1
        return param



def get_interleaving(layer: ConvolutionLayerSparseChisel, method="opt"):

    match method:
        case "opt":

            # get the cycles per bin
            cycles_per_bin = np.ceil(np.flip(np.arange(math.prod(layer.kernel_size) + 1))/layer.fine)
            if isinstance(layer, ConvolutionLayerSparseChisel):
                cycles_per_bin[-1] = 1

            # Multiply the cycles per bin by the probability of each number of non-zeros,
            # sum up the cycles and calculate the rate accordingly
            rate_per_channel = 1 / np.sum(cycles_per_bin*layer.sparsity, axis = 1)

            #Balance the channels according to their rates
            indices = np.argsort(rate_per_channel)
            indices = np.reshape(indices, (layer.channels_in()//layer.streams_in(), layer.streams_in()))
            indices[1::2, :] = indices[1::2, ::-1] # reverse every other row
            indices = indices.flatten()

            # return the indices
            return indices

        case "naive":
            return list(range(layer.channels_in()))

        case _:
            raise ValueError(f"method {method} not supported!")

def get_interleaving(layer: ConvolutionLayerSparsePointwiseChisel, method="opt"):

    match method:
        case "opt":

            #Balance the channels according to their sparsity
            indices = np.argsort(layer.channel_sparsity)
            indices = np.reshape(indices,
                    (layer.channels_in()//layer.streams_in(), layer.streams_in()))
            indices[1::2, :] = indices[1::2, ::-1] # reverse every other row
            indices = indices.flatten()

            # return the indices
            return indices

        case "naive":
            return np.arange(layer.channels_in())
        case _:
            raise ValueError(f"method {method} not supported!")

def get_clustering(layer: ConvolutionLayerSparsePointwiseChisel, method="naive"):

    match method:
        case "naive":

            # get an index for each stream in
            indices = np.arange(layer.streams_in())

            # return the reshaped indices
            return np.reshape(indices, (layer.clusters, -1))

        case _:
            raise ValueError(f"method {method} not supported!")

