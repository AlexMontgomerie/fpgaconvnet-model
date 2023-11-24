import importlib
import math
from typing import Union, List
from dataclasses import dataclass, field

from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

@dataclass(kw_only=True)
class DoubleBufferedTrait:

    def get_weight_memory_depth(self) -> int:
        return 2*super().get_weight_memory_depth()

@dataclass(kw_only=True)
class StreamingWeightsTrait:

    stream_weights: int = 0

    def get_weight_resources(self) -> (int, int):

        weight_array_depth = self.get_weight_memory_depth()
        weight_array_width = self.weight_t.width*self.fine*self.coarse_in*self.coarse_out*self.coarse_group
        weight_array_num =1

        if self.use_uram:
            weights_uram_usage = uram_array_resource_model(weight_array_depth, weight_array_width) * weight_array_num
            weights_uram_usage -= self.stream_weights
            self.weights_ram_usage = weights_uram_usage
            weights_bram_usage = 0
            if weights_uram_usage + self.stream_weights > 0:
                uram_details = uram_array_resource_model(weight_array_depth, weight_array_width, detailed=True)
                self.weight_array_unit_depth = uram_details[3]
                self.weight_array_unit_width = uram_details[1]
                if self.stream_weights > 0:
                    weights_uram_usage += self.stream_buffer()
        else:
            weights_bram_usage = bram_array_resource_model(weight_array_depth, weight_array_width, "memory") * weight_array_num
            weights_bram_usage -= self.stream_weights
            self.weights_ram_usage = weights_bram_usage
            weights_uram_usage = 0
            if weights_bram_usage + self.stream_weights > 0:
                bram_details = bram_array_resource_model(weight_array_depth, weight_array_width, "memory", detailed=True)
                self.weight_array_unit_depth = bram_details[3]
                self.weight_array_unit_width = bram_details[1]
                if self.stream_weights > 0:
                    weights_bram_usage += self.stream_buffer()

        assert self.weights_ram_usage + self.stream_weights == \
            math.ceil(weight_array_depth/self.weight_array_unit_depth) \
            * math.ceil(weight_array_width/self.weight_array_unit_width) \
            * weight_array_num

        return weights_bram_usage, weights_uram_usage
