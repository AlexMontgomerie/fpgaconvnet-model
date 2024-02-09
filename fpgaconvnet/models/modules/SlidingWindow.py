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

from fpgaconvnet.models.modules import int2bits, Module, MODULE_FONTSIZE
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, queue_lutram_resource_model

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
    backend: str = "chisel"
    regression_model: str = "linear_regression"
    streams: int = 1

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

        self.data_packing = (self.backend == "chisel")

        # perform basic module initialisation
        Module.__post_init__(self)

    def rows_out(self):
        return math.ceil((self.rows_in()-self.kernel_size[0]+self.pad_top+self.pad_bottom+1)/self.stride[0])

    def cols_out(self):
        return math.ceil((self.cols_in()-self.kernel_size[1]+self.pad_left+self.pad_right+1)/self.stride[1])

    def rate_in(self):
        padded_size = (self.rows_in()+self.pad_top+self.pad_bottom)*(self.cols_in()+self.pad_left+self.pad_right)
        size = (self.rows_in()*self.cols_in())
        return size/float(padded_size)

    def rate_out(self):
        return (self.rows_out()*self.cols_out())/float(self.rows*self.cols)

    def pipeline_depth(self):
        # return (self.cols+self.pad_left+self.pad_right)*(self.channels)*(self.kernel_size[0]-1)+\
        #         self.channels*(self.kernel_size[1]-1)
        return (self.kernel_size[0]-1)*(self.cols+self.pad_left+self.pad_right)*self.channels + \
                (self.kernel_size[1]-1)*self.channels - \
                ( self.pad_top * self.cols * self.channels + \
                self.pad_left * self.channels )
        # return self.cols*(self.channels)*(self.kernel_size[0]-1)+\
        #         self.channels*(self.kernel_size[1]-1)

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

    def buffer_estimate(self):
        # line buffer
        self.line_buffer_depth = self.channels * \
                            (self.cols + self.pad_left + self.pad_right)
        self.line_buffer_width = (self.kernel_size[0]-1) * self.streams*self.data_width
        self.line_buffer_bram = bram_array_resource_model(
                    self.line_buffer_depth, self.line_buffer_width, "fifo")
        if self.line_buffer_bram == 0:
            self.line_buffer_lutram = queue_lutram_resource_model(
                    self.line_buffer_depth, self.line_buffer_width)
        else:
            self.line_buffer_lutram = 0

        # window buffer
        self.window_buffer_depth = self.channels
        self.window_buffer_width = (self.kernel_size[1]-1) * self.streams * self.data_width
        self.window_buffer_bram = self.kernel_size[0] * bram_array_resource_model(
                    self.window_buffer_depth, self.window_buffer_width, "fifo")
        if self.window_buffer_bram == 0:
            self.window_buffer_lutram = self.kernel_size[0] * queue_lutram_resource_model(
                        self.window_buffer_depth, self.window_buffer_width)
        else:
            self.window_buffer_lutram = 0

        # frame cache
        self.frame_buff_depth = 2
        self.frame_buff_width = self.streams * self.data_width
        self.frame_buffer_bram = self.kernel_size[0] * self.kernel_size[1] * bram_array_resource_model(
                       self. frame_buff_depth, self.frame_buff_width, "fifo")
        if self.frame_buffer_bram == 0:
            self.frame_buffer_lutram = self.kernel_size[0] * self.kernel_size[1] * \
                queue_lutram_resource_model(
                self.frame_buff_depth, self.frame_buff_width)
        else:
            self.frame_buffer_lutram = 0

    def utilisation_model(self):
        # get the buffer estimates
        self.buffer_estimate()
        if self.backend == "hls":
            pass # TODO
        elif self.backend == "chisel":
            return {
                "Logic_LUT" : np.array([
                    self.streams*self.data_width,
                    (self.kernel_size[0]-1),
                    self.kernel_size[0]*(self.kernel_size[1]-1),
                    self.streams*self.data_width*(self.kernel_size[0]-1),
                    self.streams*self.data_width*self.kernel_size[0]*(self.kernel_size[1]-1),
                ]),
                "LUT_RAM"   : np.array([
                    self.line_buffer_lutram, # line buffer
                    self.window_buffer_lutram, # window buffer
                    self.frame_buffer_lutram, # frame buffer
                ]),
                "LUT_SR"    : np.array([1]),
                "FF"        : np.array([
                    int2bits(self.rows), # row_cntr
                    int2bits(self.cols), # col_cntr
                    int2bits(self.channels), # channel_cntr
                    self.streams*self.data_width, # input buffer
                    self.streams*self.data_width*self.kernel_size[0]*self.kernel_size[1], # output buffer (data)
                    self.kernel_size[0]*self.kernel_size[1], # output buffer (valid)
                    self.streams*self.data_width*(self.kernel_size[0]-1), # line buffer
                    self.streams*self.data_width*self.kernel_size[0]*(self.kernel_size[1]-1), # window buffer
                    self.kernel_size[0]*self.kernel_size[1], # frame buffer
                    1,
                ]),
                "DSP"       : np.array([0]),
                "BRAM36"    : np.array([0]),
                "BRAM18"    : np.array([0]),
            }

    def get_pred_array(self):
        return np.array([
        self.data_width, self.data_width//2,
        self.channels, self.rows, self.cols,
        self.kernel_size[0], self.kernel_size[1],
        self.stride[0], self.stride[1],
        ]).reshape(1,-1)

    def rsc(self,coef=None, model=None):

        # get the linear model estimation
        rsc = Module.rsc(self, coef, model)

        if self.regression_model == "linear_regression":
            # get the buffer estimates
            self.buffer_estimate()

            # add the bram estimation
            rsc["BRAM"] = self.line_buffer_bram + self.window_buffer_bram

            # ensure zero DSPs
            rsc["DSP"] = 0

            # correct the LUTRAM estimate
            rsc["LUT"] = rsc["LUT"] - rsc["LUT_RAM"] + self.line_buffer_lutram +\
                    self.window_buffer_lutram + self.frame_buffer_lutram

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

