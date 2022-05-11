"""
The Sliding Window module creates sequential windows of the
incoming feature map. This module allows for efficient use
of the on-chip memory compared to full featuremap caching,
with only the required number of pixels buffered. This
stream of feature map windows is used for the convolution
and pooling functions.

.. figure:: ../../../figures/sliding_window_diagram.png
"""

import math
import os
import sys
from typing import Union, List
from dataclasses import dataclass, field

import numpy as np
import pydot

from fpgaconvnet.models.modules import Module, MODULE_FONTSIZE
from fpgaconvnet.tools.resource_model import bram_memory_resource_model, bram_stream_resource_model

@dataclass
class SlidingWindow(Module):
    """
    Sliding window hardware model class.

    Attributes
    ----------
    kernel_size: int
        kernel size of the convolution layer.
    stride: int
        both row and column stride of the convolution layer.
    pad_top: int
        zero padding for the top of the featuremap.
    pad_right: int
        zero padding for the right of the featuremap.
    pad_bottom: int
        zero padding for the bottom of the featuremap.
    pad_left: int
        zero padding for the left of the featuremap.
    rows: int
        row dimension of input featuremap
    cols: int
        column dimension of input featuremap
    channels: int
        channel dimension of input featuremap
    data_width: int
        bitwidth of featuremap pixels (default is 16)
    rsc_coef: list
        list of resource model coefficients. Corresponds
        to `LUT`, `BRAM`, `DSP` and `FF` resources in
        that order.
    """
    kernel_size: Union[List[int],int]
    stride: Union[List[int],int]
    pad_top: int
    pad_right: int
    pad_bottom: int
    pad_left: int

    def __post_init__(self):
        # format kernel size as a 2 element list
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size, self.kernel_size]
        elif isinstance(self.kernel_size, list):
            assert len(self.kernel_size) == 2, "Must specify two kernel dimensions"
        else:
            raise TypeError

        # format stride as a 2 element list
        if isinstance(self.stride, int):
            self.stride = [self.stride, self.stride]
        elif isinstance(self.stride, list):
            assert len(self.stride) == 2, "Must specify two stride dimensions"
        else:
            raise TypeError

        # load the resource model coefficients
        self.rsc_coef["LUT"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/sliding_window_lut.npy"))
        self.rsc_coef["FF"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/sliding_window_ff.npy"))
        self.rsc_coef["BRAM"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/sliding_window_bram.npy"))
        self.rsc_coef["DSP"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/sliding_window_dsp.npy"))

    def utilisation_model(self):
        return {
            "LUT"  : np.array([self.data_width*self.kernel_size[0]*self.kernel_size[1],
                self.kernel_size[0]*(self.kernel_size[1]-1)*(self.data_width+math.floor(math.log(self.channels,2))),
                (self.kernel_size[1]-1)*(self.data_width+math.floor(math.log(self.channels*self.cols,2))),
                (self.kernel_size[0])*(self.kernel_size[1]-1), (self.kernel_size[1]-1)]),
            "FF"   : np.array([self.data_width*self.kernel_size[0]*self.kernel_size[1],
                self.kernel_size[0]*(self.kernel_size[1]-1)*(self.data_width+math.floor(math.log(self.channels,2))),
                (self.kernel_size[1]-1)*(self.data_width+math.floor(math.log(self.channels*self.cols,2))),
                (self.kernel_size[0])*(self.kernel_size[1]-1), (self.kernel_size[1]-1)]),
            "DSP"  : np.array([1]),
            "BRAM" : np.array([1]),
        }

    def rows_out(self):
        return int((self.rows_in()-self.kernel_size[0]+self.pad_top+self.pad_bottom)/self.stride[0]+1)

    def cols_out(self):
        return int((self.cols_in()-self.kernel_size[1]+self.pad_left+self.pad_right)/self.stride[1]+1)

    def rate_in(self):
        return 1.0 # TODO: maybe need to reduce for padding effect

    def rate_out(self):
        return (self.rows_out()*self.cols_out())/float(self.rows*self.cols)

    def pipeline_depth(self):
        return (self.cols+self.pad_left+self.pad_right)*(self.channels)*(self.kernel_size[0]-1)+self.channels*self.kernel_size[0]*(self.kernel_size[1]-1)

    def wait_depth(self):
        """
        Number of cycles delay before the first pixel is
        consumed by the module from the start signal.

        Returns
        -------
        int
        """
        return (self.pad_bottom*self.channels*self.cols+self.pad_left*self.channels+1)

    def module_info(self):
        # get the base module fields
        info = Module.module_info(self)
        # add module-specific info fields
        info["kernel_size"] = self.kernel_size
        info["stride"] = self.stride
        info["pad_top"] = self.pad_top
        info["pad_right"] = self.pad_right
        info["pad_bottom"] = self.pad_bottom
        info["pad_left"] = self.pad_left
        # return the info
        return info

    def rsc(self, coef=None):
        """
        the main resources are from the line and frame buffers.
        These use `BRAM` fifos.

        Returns
        -------
        dict
            estimated resource usage of the module. Uses the
            resource coefficients for the estimate.
        """
        # use module resource coefficients if none are given
        if coef == None:
            coef = self.rsc_coef
        # get the line buffer BRAM estimate
        line_buffer_depth = (self.cols+self.pad_left+self.pad_right)*self.channels+1
        line_buffer_bram = (self.kernel_size[0]-1) * bram_stream_resource_model(line_buffer_depth, self.data_width)
        # get the window buffer BRAM estimate
        window_buffer_depth = self.channels+1
        window_buffer_bram = self.kernel_size[0]*(self.kernel_size[1]-1) * bram_stream_resource_model(window_buffer_depth, self.data_width)
        # get the linear model estimation
        rsc = Module.rsc(self,coef)
        # add the bram estimation
        rsc["BRAM"] = line_buffer_bram + window_buffer_bram
        # return the resource usage
        return rsc

    def visualise(self, name):
        return pydot.Node(name,label="slwin", shape="box",
                height=self.kernel_size[0],
                width=self.cols*self.channels*0.1,
                style="filled", fillcolor="aquamarine",
                fontsize=MODULE_FONTSIZE)

    def functional_model(self, data):
        # check input dimensionality
        batch_size = data.shape[0]
        assert data.shape[1] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[2] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[3] == self.channels, "ERROR: invalid channel dimension"

        #pad input
        data_padded = np.ndarray((
            batch_size,
            self.rows + self.pad_bottom + self.pad_top,
            self.cols + self.pad_left   + self.pad_right,
            self.channels),dtype=float)

        for index,_ in np.ndenumerate(data_padded):
            if  (index[1] < self.pad_bottom):
                data_padded[index] = 0
            elif(index[2] < self.pad_left):
                data_padded[index] = 0
            elif(index[1] > self.rows - 1 + self.pad_bottom):
                data_padded[index] = 0
            elif(index[2] > self.cols - 1 + self.pad_left):
                data_padded[index] = 0
            else:
                data_padded[index] = data[
                    index[0],
                    index[1]-self.pad_left,
                    index[2]-self.pad_bottom,
                    index[3]]

        out = np.ndarray((
            batch_size,
            self.rows_out(),
            self.cols_out(),
            self.channels,
            self.kernel_size[0],
            self.kernel_size[1]),dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = data_padded[
                index[0],
                index[1]*self.stride[0]+index[4],
                index[2]*self.stride[1]+index[5],
                index[3]]

        return out

