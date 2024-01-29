from typing import ClassVar, Optional
from dataclasses import dataclass
from collections import OrderedDict
import math

import pydot
import numpy as np
from dacite import from_dict
import networkx as nx

from fpgaconvnet.models.layers.inner_product import InnerProductLayerBase
from fpgaconvnet.models.modules import ModuleBase
from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY
from fpgaconvnet.models.layers.utils import get_factors

@dataclass(kw_only=True)
class InnerProductLayerChiselMixin(InnerProductLayerBase):

    backend: ClassVar[BACKEND] = BACKEND.CHISEL

    @property
    def module_lookup(self) -> dict:
        return OrderedDict({
            "fork": self.get_fork_parameters,
            "vector_dot": self.get_vector_dot_parameters,
            "accum": self.get_accum_parameters,
            "glue": self.get_glue_parameters,
            "bias": self.get_bias_parameters
        })

    def get_fork_parameters(self):

        return {
            "repetitions": self.channels//self.streams_in(),
            "streams": self.streams_in(),
            "fine": 1,
            "coarse": self.coarse_out,
            "data_t": self.input_t,
        }

    def get_vector_dot_parameters(self):

        return {
            "repetitions": 1,
            "streams": self.coarse_in*self.coarse_out,
            "filters": self.filters//self.streams_out(),
            "fine": 1,
            "data_t": self.input_t,
            "weight_t": self.weight_t,
            "acc_t": self.acc_t,
        }

    def get_accum_parameters(self):

        return {
            "repetitions": 1,
            "streams": self.coarse_in*self.coarse_out,
            "channels": self.channels//self.streams_in(),
            "filters": self.filters//self.streams_out(),
            "data_t": self.acc_t,
        }

    def get_glue_parameters(self):

        return {
            "repetitions": self.filters//self.streams_out(),
            "coarse": self.coarse_in,
            "streams": self.streams_out(),
            "data_t": self.acc_t,
        }

    def get_bias_parameters(self):

        return {
            "repetitions": 1,
            "streams": self.streams_out(),
            "channels": self.filters//self.streams_out(),
            "data_t": self.output_t,
        }


    def build_module_graph(self) -> nx.DiGraph:

        # get the module graph
        self.graph = nx.DiGraph()

        # add the modules
        self.graph.add_node("fork", module=self.modules["fork"])
        self.graph.add_node("vector_dot", module=self.modules["vector_dot"])
        self.graph.add_node("accum", module=self.modules["accum"])
        self.graph.add_node("glue", module=self.modules["glue"])
        self.graph.add_node("bias", module=self.modules["bias"])

        # connect the modules
        self.graph.add_edge("fork", "vector_dot")
        self.graph.add_edge("vector_dot", "accum")
        self.graph.add_edge("accum", "glue")
        self.graph.add_edge("glue", "bias")

@dataclass(kw_only=True)
class InnerProductLayerHLSMixin(InnerProductLayerBase):

    backend: ClassVar[BACKEND] = BACKEND.HLS

    @property
    def module_lookup(self) -> dict:
        return OrderedDict({
            "fork": self.get_fork_parameters,
            "conv": self.get_conv_parameters,
            "accum": self.get_accum_parameters,
            "glue": self.get_glue_parameters,
            "bias": self.get_bias_parameters
        })

    def get_fork_parameters(self):

        return {
            **self.output_shape_dict(),
            "channels": self.channels//self.streams_in(),
            "kernel_size": [1,1],
            "coarse": self.coarse_out,
            "data_t": self.input_t,
        }

    def get_conv_parameters(self):

        return {
            **self.output_shape_dict(),
            "channels": self.channels//self.streams_in(),
            "filters": self.filters//self.streams_out(),
            "groups": 1,
            "kernel_size": [1,1],
            "fine": 1,
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
            "groups": 1,
            "data_t": self.acc_t,
        }

    def get_glue_parameters(self):

        return {
            **self.output_shape_dict(),
            "channels": self.channels//self.streams_in(),
            "filters": self.filters//self.streams_out(),
            "coarse_in": self.coarse_in,
            "coarse_out": self.coarse_out,
            "coarse_group": 1,
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
        self.graph = nx.DiGraph()

        # add the modules
        for i in range(self.streams_in()):
            self.graph.add_node("fork_{i}", module=self.modules["fork"])
            for j in range(self.coarse_out):
                self.graph.add_node("conv_{i}_{j}", module=self.modules["conv"])
                self.graph.add_node("accum_{i}_{j}", module=self.modules["accum"])
        self.graph.add_node("glue", module=self.modules["glue"])
        for j in range(self.streams_out()):
            self.graph.add_node("bias_{j}", module=self.modules["bias"])

        # connect the modules
        for i in range(self.streams_in()):
            for j in range(self.coarse_out):
                self.graph.add_edge("fork_{i}", "conv_{i}_{j}")
                self.graph.add_edge("conv_{i}_{j}", "accum_{i}_{j}")
                self.graph.add_edge("accum_{i}_{j}", "glue")
        for j in range(self.streams_out()):
            self.graph.add_edge("glue", "bias_{j}")


