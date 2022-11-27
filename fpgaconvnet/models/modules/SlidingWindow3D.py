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

from fpgaconvnet.models.modules import Module3D, MODULE_3D_FONTSIZE
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

    def __post_init__(self):
        # load the resource model coefficients
        # TODO: Update resource model coefficients FIXME -> Remove completely
        self.rsc_coef["LUT"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/sliding_window3d_lut.npy"))
        self.rsc_coef["FF"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/sliding_window3d_ff.npy"))
        self.rsc_coef["BRAM"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/sliding_window3d_bram.npy"))
        self.rsc_coef["DSP"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/sliding_window3d_dsp.npy"))

    def utilisation_model(self):
        pass
        # TODO: Update utilisation model FIXME
        return {
            "LUT"  : np.array([self.data_width*self.kernel_rows*self.kernel_cols,
                self.kernel_rows*(self.kernel_cols-1)*(self.data_width+math.floor(math.log(self.channels,2))),
                (self.kernel_cols-1)*(self.data_width+math.floor(math.log(self.channels*self.cols,2))),
                (self.kernel_rows)*(self.kernel_cols-1), (self.kernel_cols-1)]),
            "FF"   : np.array([self.data_width*self.kernel_rows*self.kernel_cols,
                self.kernel_rows*(self.kernel_cols-1)*(self.data_width+math.floor(math.log(self.channels,2))),
                (self.kernel_cols-1)*(self.data_width+math.floor(math.log(self.channels*self.cols,2))),
                (self.kernel_rows)*(self.kernel_cols-1), (self.kernel_cols-1)]),
            "DSP"  : np.array([1]),
            "BRAM" : np.array([1]),
        }

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
        pass
        return (self.cols+self.pad_left+self.pad_right)*(self.channels)*(self.kernel_rows-1)+self.channels*self.kernel_rows*(self.kernel_cols-1)

    def wait_depth(self):
        pass
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

    def rsc(self, coef=None):
        pass
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
        line_buffer_bram = (self.kernel_rows-1) * bram_stream_resource_model(line_buffer_depth, self.data_width)
        # get the window buffer BRAM estimate
        window_buffer_depth = self.channels+1
        window_buffer_bram = self.kernel_rows*(self.kernel_cols-1) * bram_stream_resource_model(window_buffer_depth, self.data_width)
        # get the linear model estimation
        rsc = Module3D.rsc(self,coef)
        # add the bram estimation
        rsc["BRAM"] = line_buffer_bram + window_buffer_bram
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

