import importlib
import math
from typing import Union, List
from dataclasses import dataclass, field

from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

@dataclass(kw_only=True)
class ConvolutionLayerDataPacking:

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        # check that the architecture is a chisel architecture
        # assert self.arch.backend == BACKEND.CHISEL

    def update_modules(self):

        # iterate over the modules
        for module in self.modules:
            match module:
                case "sliding_window" | "squeeze" | "fork":
                    self.modules[module].streams = self.streams_in()
                case "conv" | "vector_dot" | "accum":
                    self.modules[module].streams = self.coarse_in*self.coarse_out*self.coarse_group
                case "bias":
                    self.modules[module].streams = self.streams_out()
                case "glue":
                    self.modules[module].streams = 1

    def get_weight_resources(self) -> (int, int):

        # get the depth for the weights memory
        weight_memory_depth = self.get_weight_memory_depth()

        # return the memory resource model
        return self.stream_rsc(weight_memory_depth, self.fine*self.coarse_in*self.coarse_out*\
                self.coarse_group*self.weight_t.width, 1)

def get_weight_resources(self) -> (int, int):

    # get the depth for the weights memory
        weight_memory_depth = self.get_weight_memory_depth()

        bram_rsc = bram_array_resource_model(weight_memory_depth, self.fine*self.coarse_in*\
                self.coarse_out*self.coarse_group*self.weight_t.width, "memory")

        # return the memory resource model
        return bram_rsc, 0  # (bram usage, uram usage)

@dataclass(kw_only=True)
class ConvolutionLayerDoubleBuffered:

    def get_weight_memory_depth(self) -> int:
        return 2*super().get_weight_memory_depth()

@dataclass(kw_only=True)
class ConvolutionLayerStreamWeights: # TODO
    pass

@dataclass(kw_only=True)
class ConvolutionLayerStreamInputs: # TODO
    pass

