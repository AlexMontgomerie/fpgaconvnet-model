"""
The Fork module provides functionality for
parallelism within layers. By duplicating the
streams, it can be used for exploiting
parallelism across filters in the Convolution
layers.

.. figure:: ../../../figures/fork_diagram.png
"""

import math
import os
import sys
from typing import Union, List
from dataclasses import dataclass, field

import numpy as np
import pydot

from fpgaconvnet.models.modules import Module, MODULE_FONTSIZE

@dataclass
class Fork(Module):
    kernel_size: Union[List[int],int]
    coarse: int

    def __post_init__(self):

        # format kernel size as a 2 element list
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size, self.kernel_size]
        elif isinstance(self.kernel_size, list):
            assert len(self.kernel_size) == 2, "Must specify two kernel dimensions"
        else:
            raise TypeError

        # perform basic module initialisation
        Module.__post_init__(self)

    def module_info(self):
        # get the base module fields
        info = Module.module_info(self)
        # add module-specific info fields
        info["coarse"] = self.coarse
        info["kernel_size"] = self.kernel_size
        # return the info
        return info

    def utilisation_model(self):
        if self.backend == "hls":
            pass # TODO
        elif self.backend == "chisel":
            return {
                "Logic_LUT" : np.array([
                    self.streams*np.prod(self.kernel_size), # input buffer
                    self.streams*self.data_width*np.prod(self.kernel_size), # input buffer
                    self.streams*np.prod(self.kernel_size)*self.coarse, # output buffer
                    self.streams*self.data_width*np.prod(self.kernel_size)*self.coarse, # output buffer

#                     self.kernel_size[0]*self.kernel_size[1]*self.coarse, # output buffer valid
#                     self.kernel_size[0]*self.kernel_size[1]*self.kernel_size[0]*self.kernel_size[1], # input buffer ready
#                     self.data_width*self.kernel_size[0]*self.kernel_size[1], # input buffer
#                     self.data_width*self.kernel_size[0]*self.kernel_size[1]*self.coarse, # output buffer
#                     self.kernel_size[0]*self.kernel_size[1], # input buffer valid
#                     self.kernel_size[0]*self.kernel_size[1]*self.coarse, # output buffer ready
                    1,
                ]),
                "LUT_RAM"   : np.array([0]),
                "LUT_SR"    : np.array([0]),
                "FF"    : np.array([
                    self.streams*np.prod(self.kernel_size), # input buffer (ready)
                    self.streams*self.data_width*np.prod(self.kernel_size)*self.coarse, # output buffer (data)
                    # self.kernel_size[0]*self.kernel_size[1]*self.coarse, # output buffer (valid)
                    1,
                ]),
                "DSP"       : np.array([0]),
                "BRAM36"    : np.array([0]),
                "BRAM18"    : np.array([0]),
            }
        else:
            raise ValueError(f"{self.backend} backend not supported")

    def get_pred_array(self):
        return np.array([
        self.data_width, self.data_width//2,
        self.coarse, np.prod(self.kernel_size), self.streams
        ]).reshape(1,-1)

    def visualise(self, name):
        return pydot.Node(name,label="fork", shape="box",
                style="filled", fillcolor="azure",
                fontsize=MODULE_FONTSIZE)

    def functional_model(self, data):

        # replicate for coarse streams
        return np.repeat(np.expand_dims(data, axis=-2), self.coarse, axis=-2)


