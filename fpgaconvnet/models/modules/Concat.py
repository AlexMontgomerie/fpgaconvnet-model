"""
"""

import math
import os
import sys
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pydot

from fpgaconvnet.models.modules import Module

@dataclass
class Concat(Module):
    channels: List[int]
    ports_in: int
    biases_width: int = field(default=16, init=False)

    def __post_init__(self):
        pass

    def __post_init__(self):
        pass

    def channels_in(self, port_index=0):
        return self.channels[port_index]

    def channels_out(self, port_index=0):
        return sum(self.channels)

    def rate_in(self, port_index=0):
        return self.channels_in(port_index)/float(sum(self.channels))

    def module_info(self):
        return {
            'type'      : self.__class__.__name__.upper(),
            'rows'      : self.rows_in(),
            'cols'      : self.cols_in(),
            'channels'  : self.channels,
            'ports_in'      : self.ports_in,
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out()
        }

    def functional_model(self, data):
        # check input dimensionality
        assert len(data) == self.ports_in , "ERROR: invalid row dimension"

        # concatenate along the channel dimension
        return np.concatenate(data, axis=-2)


