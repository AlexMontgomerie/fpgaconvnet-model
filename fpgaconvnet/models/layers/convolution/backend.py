from typing import ClassVar, Optional
from dataclasses import dataclass
from collections import OrderedDict
import math

import pydot
import numpy as np
from dacite import from_dict
import networkx as nx

from fpgaconvnet.models.layers.convolution import ConvolutionLayerBase
from fpgaconvnet.models.modules import ModuleBase
from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY
from fpgaconvnet.models.layers.utils import get_factors

@dataclass(kw_only=True)
class ConvolutionLayerChiselMixin(ConvolutionLayerBase):

    backend: ClassVar[BACKEND] = BACKEND.CHISEL

    @property
    def module_lookup(self) -> dict:
        return OrderedDict({
            "pad": self.get_pad_parameters,
            "sliding_window": self.get_sliding_window_parameters,
            "squeeze": self.get_squeeze_parameters,
            "fork": self.get_fork_parameters,
            "vector_dot": self.get_vector_dot_parameters,
            "accum": self.get_accum_parameters,
            "glue": self.get_glue_parameters,
            "bias": self.get_bias_parameters
        })

    def get_pad_parameters(self):

        return {
            "repetitions": 1,
            "streams": self.streams_in(),
            **self.input_shape_dict(),
            "channels": self.channels//self.streams_in(),
            "pad_top": self.pad_top,
            "pad_right": self.pad_right,
            "pad_bottom": self.pad_bottom,
            "pad_left": self.pad_left,
            "data_t": self.input_t,
        }

    def get_sliding_window_parameters(self):

        # get the dimensions from the pad module
        rows, cols, channels = self.modules["pad"].output_iter_space[0] # TODO: make 3D work with this aswell

        return {
            "repetitions": 1,
            "streams": self.streams_in(),
            "rows": rows,
            "cols": cols,
            "channels": channels//self.streams_in(),
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "data_t": self.input_t,
        }

    def get_squeeze_parameters(self):

        return {
            "repetitions": math.prod(self.output_shape()[:-1])*self.channels//self.streams_in(),
            "streams": self.streams_in(),
            "coarse_in": math.prod(self.kernel_size),
            "coarse_out": self.fine,
            "data_t": self.input_t,
        }

    def get_fork_parameters(self):

        return {
            "repetitions": math.prod(self.output_shape()[:-1])*self.channels//self.streams_in(),
            "streams": self.streams_in(),
            "fine": self.fine,
            "coarse": self.coarse_out,
            "data_t": self.input_t,
        }

    def get_vector_dot_parameters(self):

        repetitions = math.prod(self.output_shape()[:-1]) * \
                self.channels//self.streams_in() * \
                math.prod(self.kernel_size) // self.fine

        return {
            "repetitions": repetitions,
            "streams": self.coarse_in*self.coarse_out*self.coarse_group,
            "filters": self.filters//self.streams_out(),
            "fine": self.fine,
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
            "channels": channels,
            "filters": self.filters//self.streams_out(),
            "data_t": self.acc_t,
        }

    def get_glue_parameters(self):

        return {
            "repetitions": math.prod(self.output_shape())//self.streams_out(),
            "coarse": self.coarse_in,
            "streams": self.streams_out(),
            "data_t": self.acc_t,
        }

    def get_bias_parameters(self):

        return {
            "repetitions": math.prod(self.output_shape()[:-1]),
            "streams": self.streams_out(),
            "channels": self.filters//self.streams_out(),
            "data_t": self.output_t,
        }


    def build_module_graph(self) -> nx.DiGraph:

        # get the module graph
        self.module_graph = nx.DiGraph()

        # add the modules
        self.module_graph.add_node("pad", module=self.modules["pad"])
        self.module_graph.add_node("sliding_window", module=self.modules["sliding_window"])
        self.module_graph.add_node("squeeze", module=self.modules["squeeze"])
        self.module_graph.add_node("fork", module=self.modules["fork"])
        self.module_graph.add_node("vector_dot", module=self.modules["vector_dot"])
        self.module_graph.add_node("accum", module=self.modules["accum"])
        self.module_graph.add_node("glue", module=self.modules["glue"])
        self.module_graph.add_node("bias", module=self.modules["bias"])

        # connect the modules
        self.module_graph.add_edge("pad", "sliding_window")
        self.module_graph.add_edge("sliding_window", "squeeze")
        self.module_graph.add_edge("squeeze", "fork")
        self.module_graph.add_edge("fork", "vector_dot")
        self.module_graph.add_edge("vector_dot", "accum")
        self.module_graph.add_edge("accum", "glue")
        self.module_graph.add_edge("glue", "bias")

    def get_fine_feasible(self):
        return get_factors(np.prod(self.kernel_size))

@dataclass(kw_only=True)
class ConvolutionLayerHLSMixin(ConvolutionLayerBase):

    backend: ClassVar[BACKEND] = BACKEND.HLS

    @property
    def module_lookup(self) -> dict:
        return OrderedDict({
            "sliding_window": self.get_sliding_window_parameters,
            "fork": self.get_fork_parameters,
            "conv": self.get_conv_parameters,
            "accum": self.get_accum_parameters,
            "glue": self.get_glue_parameters,
            "bias": self.get_bias_parameters
        })

    def get_sliding_window_parameters(self):

        return {
            **self.input_shape_dict(),
            "channels": self.channels//self.streams_in(),
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "pad": self.pad,
            "data_t": self.input_t,
        }

    def get_fork_parameters(self):

        return {
            **self.output_shape_dict(),
            "channels": self.channels//self.streams_in(),
            "kernel_size": self.kernel_size,
            "coarse": self.coarse_out,
            "data_t": self.input_t,
        }

    def get_conv_parameters(self):

        return {
            **self.output_shape_dict(),
            "channels": self.channels//self.streams_in(),
            "filters": self.filters//self.streams_out(),
            "groups": self.groups,
            "kernel_size": self.kernel_size,
            "fine": self.fine,
            "data_t": self.input_t,
            "weight_t": self.weight_t,
            "acc_t": self.acc_t,
        }

    def get_accum_parameters(self):

        return {
            **self.output_shape_dict(),
            "rows": self.rows_out(),
            "cols": self.cols_out(),
            "channels": self.channels//self.streams_in(),
            "filters": self.filters//self.streams_out(),
            "groups": self.groups,
            "data_t": self.acc_t,
        }

    def get_glue_parameters(self):

        return {
            **self.output_shape_dict(),
            "channels": self.channels//self.streams_in(),
            "filters": self.filters//self.streams_out(),
            "coarse_in": self.coarse_in,
            "coarse_out": self.coarse_out,
            "coarse_group": self.coarse_group,
            "data_t": self.acc_t,
        }

    def get_bias_parameters(self):

        return {
            **self.output_shape_dict(),
            "channels": self.channels//self.streams_in(),
            "filters": self.filters//self.streams_out(),
            "data_t": self.acc_t,
        }

    def build_module_graph(self) -> nx.DiGraph:

        # get the module graph
        self.module_graph = nx.DiGraph()

        # add the modules
        for i in range(self.streams_in()):
            self.module_graph.add_node("sliding_window_{i}", module=self.modules["sliding_window"])
            self.module_graph.add_node("fork_{i}", module=self.modules["fork"])
            for j in range(self.coarse_out):
                self.module_graph.add_node("conv_{i}_{j}", module=self.modules["conv"])
                self.module_graph.add_node("accum_{i}_{j}", module=self.modules["accum"])
        self.module_graph.add_node("glue", module=self.modules["glue"])
        for j in range(self.streams_out()):
            self.module_graph.add_node("bias_{j}", module=self.modules["bias"])

        # connect the modules
        for i in range(self.streams_in()):
            self.module_graph.add_edge("sliding_window_{i}", "fork_{i}")
            for j in range(self.coarse_out):
                self.module_graph.add_edge("fork_{i}", "conv_{i}_{j}")
                self.module_graph.add_edge("conv_{i}_{j}", "accum_{i}_{j}")
                self.module_graph.add_edge("accum_{i}_{j}", "glue")
        for j in range(self.streams_out()):
            self.module_graph.add_edge("glue", "bias_{j}")

    def get_fine_feasible(self):
        return [ 1, self.kernel_size[0], self.kernel_size[0]*self.kernel_size[1] ] # TODO: extend to 3D case


