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

from fpgaconvnet.models.modules import int2bits, Module3D, MODULE_3D_FONTSIZE
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, queue_lutram_resource_model

@dataclass
class SlidingWindow3D(Module3D):
    """
    Sliding window hardware model class.

    Attributes
    ----------
    kernel_rows: int
        kernel size of the rows of the convolution layer.
    kernel_cols: int
        kernel size of the columns of the convolution layer.
    kernel_depth: int
        kernel size of the depth of the convolution layer.
    stride_rows: int
        row stride of the convolution layer.
    stride_cols: int
        column stride of the convolution layer.
    stride_depth: int
        depth stride of the convolution layer.
    pad_top: int
        zero padding for the top of the featuremap.
    pad_right: int
        zero padding for the right of the featuremap.
    pad_front: int
        zero padding for the front(depth) of the featuremap.
    pad_bottom: int
        zero padding for the bottom of the featuremap.
    pad_left: int
        zero padding for the left of the featuremap.
    pad_back: int
        zero padding for the back(depth) of the featuremap.
    rows: int
        row dimension of input featuremap
    cols: int
        column dimension of input featuremap
    depth: int
        depth dimension of input featuremap
    channels: int
        channel dimension of input featuremap
    data_width: int
        bitwidth of featuremap pixels (default is 16)
    rsc_coef: list
        list of resource model coefficients. Corresponds
        to `LUT`, `BRAM`, `DSP` and `FF` resources in
        that order.
    """
    # kernel_size: Union[List[int], int]
    kernel_rows: int
    kernel_cols: int
    kernel_depth: int
    # stride: Union[List[int],int]
    stride_rows: int
    stride_cols: int
    stride_depth: int
    pad_top: int
    pad_right: int
    pad_front: int
    pad_bottom: int
    pad_left: int
    pad_back: int
    backend: str = "chisel"
    regression_model: str = "linear_regression"
    streams: int = 1

    @property
    def kernel_size(self):
        return [ self.kernel_rows, self.kernel_cols, self.kernel_depth ]

    @kernel_size.setter
    def kernel_size(self, kernel_size):
        self.kernel_rows = kernel_size[0]
        self.kernel_cols = kernel_size[1]
        self.kernel_depth = kernel_size[2]

    def rows_out(self):
        return int((self.rows_in()-self.kernel_rows+self.pad_top+self.pad_bottom)/self.stride_rows+1)

    def cols_out(self):
        return int((self.cols_in()-self.kernel_cols+self.pad_left+self.pad_right)/self.stride_cols+1)

    def depth_out(self):
        return int((self.depth_in()-self.kernel_depth+self.pad_back+self.pad_front)/self.stride_depth+1)

    def rate_in(self):
        return (self.rows_in()*self.cols_in()*self.depth_in())/float(
                (self.rows_in()+self.pad_top+self.pad_bottom)*\
                (self.cols_in()+self.pad_left+self.pad_right)*\
                (self.depth_in()+self.pad_front+self.pad_back))

    def rate_out(self):
        return (self.rows_out()*self.cols_out()*self.depth_out())/float(self.rows*self.cols*self.depth)

    def pipeline_depth(self):
        return (self.cols+self.pad_left+self.pad_right)*(self.depth+self.pad_front+self.pad_back)*self.channels*(self.kernel_rows-1) +\
               (self.depth+self.pad_front+self.pad_back)*self.channels*(self.kernel_cols-1) +\
               self.channels*(self.kernel_depth-1) - \
                ( self.pad_top * self.cols * self.depth * self.channels + \
                self.pad_front * self.cols * self.channels + \
                self.pad_left * self.channels )

    def wait_depth(self):
        """
        Number of cycles delay before the first pixel is
        consumed by the module from the start signal.

        Returns
        -------
        int
        """
        return (self.pad_bottom*self.channels*(self.cols+self.pad_left+self.pad_right)*(self.depth+self.pad_front+self.pad_back) + self.pad_left* self.channels*(self.depth+self.pad_front+self.pad_back) + self.pad_front*self.channels+1)

    def module_info(self):
        # get the base module fields
        info = Module3D.module_info(self)
        # add module-specific info fields
        # info["kernel_size"] = self.kernel_size
        info["kernel_rows"] = self.kernel_rows
        info["kernel_cols"] = self.kernel_cols
        info["kernel_depth"] = self.kernel_depth
        # info["stride"] = self.stride
        info["stride_rows"] = self.stride_rows
        info["stride_cols"] = self.stride_cols
        info["stride_depth"] = self.stride_depth
        info["pad_top"] = self.pad_top
        info["pad_right"] = self.pad_right
        info["pad_front"] = self.pad_front
        info["pad_bottom"] = self.pad_bottom
        info["pad_left"] = self.pad_left
        info["pad_back"] = self.pad_back
        # return the info
        return info

    def buffer_estimate(self):
        # line buffer
        self.line_buffer_depth = self.channels * \
                            (self.depth + self.pad_front + self.pad_back) * \
                            (self.cols + self.pad_left + self.pad_right)
        self.line_buffer_width = (self.kernel_rows-1) * self.streams * self.data_width
        self.line_buffer_bram = bram_array_resource_model(
                    self.line_buffer_depth, self.line_buffer_width, 'fifo')
        if self.line_buffer_bram == 0:
            self.line_buffer_lutram = queue_lutram_resource_model(
                    self.line_buffer_depth, self.line_buffer_width)
        else:
            self.line_buffer_lutram = 0

        # window buffer
        self.window_buffer_depth = self.channels * \
                                (self.depth + self.pad_front + self.pad_back)
        self.window_buffer_width = (self.kernel_cols-1) * self.streams * self.data_width
        self.window_buffer_bram = self.kernel_rows * bram_array_resource_model(
                self.window_buffer_depth, self.window_buffer_width, 'fifo')
        if self.window_buffer_bram == 0:
            self.window_buffer_lutram = self.kernel_rows * queue_lutram_resource_model(
                    self.window_buffer_depth, self.window_buffer_width)
        else:
            self.window_buffer_lutram = 0

        # tensor buffer
        self.tensor_buffer_depth = self.channels
        self.tensor_buffer_width = (self.kernel_depth-1) * self.streams * self.data_width
        self.tensor_buffer_bram = self.kernel_rows * self.kernel_cols * bram_array_resource_model(
                self.tensor_buffer_depth, self.tensor_buffer_width, 'fifo')
        if self.tensor_buffer_bram == 0:
            self.tensor_buffer_lutram = self.kernel_rows * self.kernel_cols * queue_lutram_resource_model(
                    self.tensor_buffer_depth, self.tensor_buffer_width)
        else:
            self.tensor_buffer_lutram = 0

        # frame cache
        self.frame_buff_depth = 2
        self.frame_buff_width = self.streams * self.data_width
        self.frame_buffer_bram = self.kernel_rows * self.kernel_cols * self.kernel_depth *\
                     bram_array_resource_model(self.frame_buff_depth, self.frame_buff_width, "fifo")
        if self.frame_buffer_bram == 0:
            self.frame_buffer_lutram = self.kernel_rows * self.kernel_cols * self.kernel_depth * \
                queue_lutram_resource_model(
                self.frame_buff_depth, self.frame_buff_width)
        else:
            self.frame_buffer_lutram = 0


    def utilisation_model(self):
        if self.backend == "hls":
            pass # TODO
        elif self.backend == "chisel":
            self.buffer_estimate()
            return {
                "Logic_LUT" : np.array([
                    self.data_width,
                    int2bits(self.rows), # row_cntr
                    int2bits(self.cols), # col_cntr
                    int2bits(self.channels), # channel_cntr
                    int2bits(self.depth), # depth cntr
                    (self.kernel_size[0]-1),
                    (self.kernel_size[0]-2), # line buffer ready
                    self.line_buffer_width,
                    self.kernel_size[0]*(self.kernel_size[1]-1),
                    self.kernel_size[0]*(self.kernel_size[1]-2), # window buffer ready
                    self.kernel_size[0], # window buffer ready
                    self.kernel_rows * self.window_buffer_width,
                    self.kernel_size[0]*self.kernel_size[1]*(self.kernel_size[2]-1),
                    self.kernel_size[0]*self.kernel_size[1]*(self.kernel_size[2]-2), # tensor buffer ready
                    self.kernel_size[0]*self.kernel_size[1], # tensor buffer ready
                    self.kernel_rows * self.kernel_cols * self.tensor_buffer_width,
                    self.kernel_size[0]*self.kernel_size[1]*self.kernel_size[2], # frame out logic
                    1,
                ]),
                "LUT_RAM"   : np.array([
                    self.line_buffer_lutram, # line buffer
                    self.window_buffer_lutram, # window buffer
                    self.tensor_buffer_lutram, # tensor buffer
                    self.frame_buffer_lutram, # frame buffer
                    0, # filter buffer, todo: remove
                    1,
                ]),
                "LUT_SR"    : np.array([1]),
                "FF"        : np.array([
                    int2bits(self.rows), # row_cntr
                    int2bits(self.cols), # col_cntr
                    int2bits(self.channels), # channel_cntr
                    int2bits(self.depth), # depth cntr
                    self.streams*self.data_width, # input buffer
                    self.streams*self.data_width*self.kernel_size[0]*self.kernel_size[1]*self.kernel_size[2], # output buffer (data)
                    self.kernel_size[0]*self.kernel_size[1]*self.kernel_size[2], # output buffer (valid)
                    self.line_buffer_width, # line buffer
                    self.kernel_rows * self.window_buffer_width, # window buffer
                    self.kernel_rows * self.kernel_cols * self.tensor_buffer_width, # tensor buffer
                    self.kernel_rows * self.kernel_cols * self.kernel_depth * self.frame_buff_width, # frame buffer
                    1,
                ]),
                "DSP"       : np.array([0]),
                "BRAM36"    : np.array([
                    self.line_buffer_width, # line buffer (width)
                    self.line_buffer_depth, # line buffer (depth)
                    self.kernel_rows * self.window_buffer_width, # window buffer (width)
                    self.window_buffer_depth, # window buffer (depth)
                    self.kernel_rows * self.kernel_cols * self.tensor_buffer_width, # tensor buffer (width)
                    self.tensor_buffer_depth,
                ]),
                "BRAM18"    : np.array([
                    self.line_buffer_width, # line buffer (width)
                    self.line_buffer_depth, # line buffer (depth)
                    self.kernel_rows * self.window_buffer_width, # window buffer (width)
                    self.window_buffer_depth, # window buffer (depth)
                    self.kernel_rows * self.kernel_cols * self.tensor_buffer_width, # tensor buffer (width)
                    self.tensor_buffer_depth,
                ]),
            }

    def get_pred_array(self):
        return np.array([
        self.data_width, self.data_width//2,
        self.channels, self.rows, self.cols,
        self.depth, self.streams,
        self.kernel_rows, self.kernel_cols,
        self.kernel_depth, self.stride_rows,
        self.stride_cols, self.stride_depth,
        ]).reshape(1,-1)

    def rsc(self,coef=None, model=None):

        # get the linear model estimation
        rsc = Module3D.rsc(self, coef, model)

        if self.regression_model == "linear_regression":
            # get the buffer estimates
            self.buffer_estimate()

            # add the bram estimation
            rsc["BRAM"] = self.line_buffer_bram + self.window_buffer_bram +\
                 self.tensor_buffer_bram + self.frame_buffer_bram

            # ensure zero DSPs
            rsc["DSP"] = 0

            # correct the LUTRAM estimate
            rsc["LUT"] = rsc["LUT"] - rsc["LUT_RAM"] + self.line_buffer_lutram +\
                    self.window_buffer_lutram + self.tensor_buffer_lutram +\
                    self.frame_buffer_lutram

        # return the resource usage
        return rsc

    def visualise(self, name):
        return pydot.Node(name,label="slwin3d", shape="box",
                height=self.kernel_rows,
                width=self.cols*self.channels*0.1,
                style="filled", fillcolor="aquamarine",
                fontsize=MODULE_3D_FONTSIZE)

    def functional_model(self, data):
        # check input dimensionality
        batch_size = data.shape[0]
        assert data.shape[1] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[2] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[3] == self.depth    , "ERROR: invalid depth dimension"
        assert data.shape[4] == self.channels, "ERROR: invalid channel dimension"

        #pad input
        data_padded = np.ndarray((
            batch_size,
            self.rows + self.pad_bottom + self.pad_top,
            self.cols + self.pad_left + self.pad_right,
            self.depth + self.pad_back + self.pad_front,
            self.channels),dtype=float)

        for index,_ in np.ndenumerate(data_padded):
            if  (index[1] < self.pad_bottom):
                data_padded[index] = 0
            elif(index[2] < self.pad_left):
                data_padded[index] = 0
            elif(index[3] < self.pad_back):
                data_padded[index] = 0
            elif(index[1] > self.rows - 1 + self.pad_bottom):
                data_padded[index] = 0
            elif(index[2] > self.cols - 1 + self.pad_left):
                data_padded[index] = 0
            elif(index[3] > self.depth - 1 + self.pad_back):
                data_padded[index] = 0
            else:
                data_padded[index] = data[
                    index[0],
                    index[1] - self.pad_left,
                    index[2] - self.pad_bottom,
                    index[3] - self.pad_back,
                    index[4]]

        out = np.ndarray((
            batch_size,
            self.rows_out(),
            self.cols_out(),
            self.depth_out(),
            self.channels,
            self.kernel_rows,
            self.kernel_cols,
            self.kernel_depth),dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = data_padded[
                index[0],
                index[1]*self.stride_rows+index[5],
                index[2]*self.stride_cols+index[6],
                index[3]*self.stride_depth+index[7],
                index[4]]

        return out

