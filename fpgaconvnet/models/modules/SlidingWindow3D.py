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
from fpgaconvnet.tools.resource_analytical_model import bram_memory_resource_model, bram_stream_resource_model

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

    def __post_init__(self):

        # get the cache path
        rsc_cache_path = os.path.dirname(__file__) + \
                f"/../../coefficients/{self.backend}"

        # iterate over resource types
        self.rsc_coef = {}
        for rsc_type in self.utilisation_model():
            # TODO Update this to use the new resource model for 3D version FIXME
            # load the resource coefficients from the 2D version
            coef_path = os.path.join(rsc_cache_path, f"{self.__class__.__name__.split('3D')[0]}_{rsc_type}.npy".lower())
            self.rsc_coef[rsc_type] = np.load(coef_path)

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
        return (self.cols+self.pad_left+self.pad_right)*(self.depth+self.pad_front+self.pad_back)*(self.channels)*(self.kernel_rows-1) +\
               (self.depth+self.pad_front+self.pad_back)*(self.channels)*(self.kernel_cols-1) +\
               self.channels*((self.kernel_rows-1)*self.kernel_cols*self.kernel_depth + (self.kernel_cols-1)*self.kernel_depth + (self.kernel_depth-1))

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

    def memory_usage(self):
        # TODO Update this to use the new resource model for 3D version FIXME
        if self.backend == "chisel":
            return self.data_width*(self.kernel_rows-1)*(self.cols*self.channels) + \
                self.data_width*self.kernel_rows*(self.kernel_cols-1)*(self.channels) + \
                self.data_width*self.kernel_rows*self.kernel_cols
        else:
            raise NotImplementedError

    def utilisation_model(self):
        if self.backend == "hls":
            pass # TODO
        elif self.backend == "chisel":
            # TODO Update this to use the new resource model for 3D version FIXME
            return {
                "Logic_LUT" : np.array([
                    self.data_width,
                    (self.kernel_rows-1),
                    self.kernel_rows*(self.kernel_cols-1),
                    (self.kernel_rows-1)*(self.cols*self.channels+1),
                    self.kernel_rows*(self.kernel_cols-1)*(self.channels+1),
                ]),
                "LUT_RAM"   : np.array([
                    self.data_width*(self.kernel_rows-1)*(self.cols*self.channels), # line buffer
                    self.data_width*self.kernel_rows*(self.kernel_cols-1)*(self.channels), # window buffer
                    self.data_width*self.kernel_rows*self.kernel_cols, # frame buffer
                ]),
                "LUT_SR"    : np.array([1]),
                "FF"        : np.array([
                    int2bits(self.rows), # row_cntr
                    int2bits(self.cols), # col_cntr
                    int2bits(self.channels), # channel_cntr
                    self.data_width, # input buffer
                    self.data_width*self.kernel_rows*self.kernel_cols, # output buffer
                    (self.kernel_rows-1)*(self.cols*self.channels), # line buffer
                    self.kernel_rows*(self.kernel_cols-1)*(self.channels), # window buffer
                    self.kernel_rows*self.kernel_cols, # frame buffer
                ]),
                "DSP"       : np.array([0]),
                "BRAM36"    : np.array([
                    self.data_width*(self.kernel_rows-1)*(self.cols*self.channels), # line buffer
                    self.data_width*self.kernel_rows*(self.kernel_cols-1)*(self.channels), # window buffer
                    self.data_width*self.kernel_rows*self.kernel_cols, # frame buffer
                ]),
                "BRAM18"    : np.array([
                    self.data_width*(self.kernel_rows-1)*(self.cols*self.channels), # line buffer
                    self.data_width*self.kernel_rows*(self.kernel_cols-1)*(self.channels), # window buffer
                    self.data_width*self.kernel_rows*self.kernel_cols, # frame buffer
                ]),
            }

    def rsc(self,coef=None):

        # use module resource coefficients if none are given
        if coef == None:
            coef = self.rsc_coef

        # get the linear model estimation
        rsc = Module3D.rsc(self, coef)

        # get the line buffer BRAM estimate
        line_buffer_depth = self.channels * (
                            (self.depth + self.pad_front + self.pad_back) *\
                            (self.cols + self.pad_left + self.pad_right) -\
                            (self.kernel_cols - 1) * self.depth -\
                            (self.kernel_depth - 1)
                            ) + 1
        line_buffer_bram = (self.kernel_rows-1) * \
                bram_stream_resource_model(line_buffer_depth, self.data_width)

        # get the window buffer BRAM estimate
        window_buffer_depth = self.channels * (
                              (self.depth + self.pad_front + self.pad_back)
                              ) + 1
        window_buffer_bram = self.kernel_rows*(self.kernel_cols-1) * \
                bram_stream_resource_model(window_buffer_depth, self.data_width)

        # get the tensor buffer BRAM estimate
        tensor_buffer_depth = self.channels + 1
        tensor_buffer_bram = self.kernel_rows*self.kernel_cols*(self.kernel_depth-1) * \
                bram_stream_resource_model(tensor_buffer_depth, self.data_width)

        # add the bram estimation
        rsc["BRAM"] = line_buffer_bram + window_buffer_bram + tensor_buffer_bram

        # ensure zero DSPs
        rsc["DSP"] = 0

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

