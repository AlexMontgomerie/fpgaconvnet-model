import numpy as np
import math
import pydot
import torch
from typing import Union, List

from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.tools.resource_analytical_model import bram_memory_resource_model
from fpgaconvnet.data_types import FixedPoint

from fpgaconvnet.models.modules import SlidingWindow
from fpgaconvnet.models.modules import VectorDot
from fpgaconvnet.models.modules import Squeeze
from fpgaconvnet.models.modules import Fork
from fpgaconvnet.models.modules import Accum
from fpgaconvnet.models.modules import Glue
from fpgaconvnet.models.modules import Bias

from fpgaconvnet.models.layers import ConvolutionLayerBase

class ConvolutionLayer(ConvolutionLayerBase):

    def __init__(
            self,
            filters: int,
            rows: int,
            cols: int,
            channels: int,
            coarse_in: int = 1,
            coarse_out: int = 1,
            coarse_group: int = 1,
            kernel_size: Union[List[int], int] = 3,
            stride: Union[List[int], int] = 1,
            groups: int = 1,
            pad: Union[List[int], int] = 0,
            fine: int  = 1,
            input_t: FixedPoint = FixedPoint(16,8),
            output_t: FixedPoint = FixedPoint(16,8),
            weight_t: FixedPoint = FixedPoint(16,8),
            acc_t: FixedPoint = FixedPoint(32,16),
            has_bias: int = 0 # default to no bias for old configs
        ):

        # initialise parent class
        super().__init__(rows, cols, channels,
                coarse_in, coarse_out)

        # save data types
        self.input_t = input_t
        self.output_t = output_t
        self.weight_t = weight_t
        self.acc_t = acc_t

        # save bias flag
        self.has_bias = has_bias

        # init variables
        self._kernel_size = self.format_kernel_size(kernel_size)
        self._stride = self.format_stride(stride)
        self._pad = self.format_pad(pad)
        self._groups = groups
        self._coarse_group = coarse_group
        self._fine = fine
        self._filters = filters

        self._pad_top = self._pad[0]
        self._pad_right = self._pad[3]
        self._pad_bottom = self._pad[2]
        self._pad_left = self._pad[1]

        # init modules
        self.modules["sliding_window"] = SlidingWindow(self.rows_in(), self.cols_in(),
                int(self.channels_in()/self.coarse_in), self.kernel_size, self.stride,
                self.pad_top, self.pad_right, self.pad_bottom, self.pad_left)
        self.modules["squeeze"] = Squeeze(self.rows_out(), self.cols_out(),
                int(self.channels_in()/self.coarse_in),
                self.kernel_size[0]*self.kernel_size[1], self.fine)
        self.modules["fork"] = Fork(self.rows_out(), self.cols_out(),
                int(self.channels_in()/self.coarse_in), self.fine, self.coarse_out)
        self.modules["vector_dot"] = VectorDot(self.rows_out(), self.cols_out(),
                int(self.channels_in()/self.coarse_in),
                int(self.filters/self.coarse_out), self.fine)
        self.modules["accum"] = Accum(self.rows_out(), self.cols_out(),
                int(self.kernel_size[0]*self.kernel_size[1]*self.channels_in()/(self.fine*self.coarse_in)),
                int(self.filters/self.coarse_out), self.groups)
        self.modules["glue"] = Glue(self.rows_out(), self.cols_out(), 1,
                int(self.filters/self.coarse_out), self.coarse_in, self.coarse_out)
        self.modules["bias"] = Bias(self.rows_out(), self.cols_out(), 1, self.filters)

        self.update()

    def update(self):
        # sliding window
        self.modules['sliding_window'].rows     = self.rows
        self.modules['sliding_window'].cols     = self.cols
        self.modules['sliding_window'].channels = self.channels//(self.coarse_in*self.coarse_group)
        self.modules['sliding_window'].data_width   = self.input_t.width
        # squeeze
        self.modules['squeeze'].rows     = self.rows_out()
        self.modules['squeeze'].cols     = self.cols_out()
        self.modules['squeeze'].channels = self.channels//(self.coarse_in*self.coarse_group)
        self.modules['squeeze'].coarse_out = self.fine
        self.modules['squeeze'].data_width = self.input_t.width
        # fork
        self.modules['fork'].rows     = self.rows_out()
        self.modules['fork'].cols     = self.cols_out()
        self.modules['fork'].channels = self.channels_in()//(self.coarse_in*self.coarse_group)
        self.modules['fork'].coarse   = self.coarse_out
        self.modules['fork'].data_width     = self.input_t.width
        # kernel dot
        self.modules['vector_dot'].rows     = self.rows_out()
        self.modules['vector_dot'].cols     = self.cols_out()
        self.modules['vector_dot'].channels = (self.channels*self.kernel_size[0]*self.kernel_size[1])//(
                self.fine*self.coarse_in*self.coarse_group)
        self.modules['vector_dot'].filters  = self.filters//(self.coarse_out*self.coarse_group)
        self.modules['vector_dot'].fine     = self.fine
        self.modules['vector_dot'].data_width     = self.input_t.width
        self.modules['vector_dot'].weight_width   = self.weight_t.width
        self.modules['vector_dot'].acc_width      = self.acc_t.width
        # accum
        self.modules['accum'].rows     = self.rows_out()
        self.modules['accum'].cols     = self.cols_out()
        self.modules['accum'].channels = (self.channels*self.kernel_size[0]*self.kernel_size[1])//(
                self.fine*self.coarse_in*self.coarse_group)
        self.modules['accum'].filters  = self.filters//(self.coarse_out*self.coarse_group)
        self.modules['accum'].groups   = self.groups//self.coarse_group
        self.modules['accum'].data_width    = self.acc_t.width
        # glue
        self.modules['glue'].rows       = self.rows_out()
        self.modules['glue'].cols       = self.cols_out()
        self.modules['glue'].filters    = self.filters//self.coarse_group
        self.modules['glue'].coarse_in  = self.coarse_in
        self.modules['glue'].coarse_out = self.coarse_out
        self.modules['glue'].data_width = self.acc_t.width
        # bias
        self.modules['bias'].rows           = self.rows_out()
        self.modules['bias'].cols           = self.cols_out()
        self.modules['bias'].filters        = self.filters
        self.modules['bias'].data_width     = self.output_t.width
        self.modules['bias'].biases_width   = self.acc_t.width

    def resource(self):

        sw_rsc          = self.modules['sliding_window'].rsc()
        fork_rsc        = self.modules['fork'].rsc()
        vector_dot_rsc  = self.modules['vector_dot'].rsc()
        accum_rsc       = self.modules['accum'].rsc()
        glue_rsc        = self.modules['glue'].rsc()
        bias_rsc        = self.modules['bias'].rsc()

        if self.kernel_size[0] == 1 and self.kernel_size[1] == 1:
            sw_rsc      = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        if self.coarse_out == 1:
            fork_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        if int(self.channels_in()/(self.coarse_in*self.coarse_group)) == 1:
            accum_rsc   = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        if self.coarse_in == 1:
            glue_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        # condition if there are no biases for the layer
        if self.has_bias:
            bias_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}

        # weight usage
        weight_memory_depth = float((self.filters/self.groups)* \
                                    self.channels_in()* \
                                    self.kernel_size[0]* \
                                    self.kernel_size[1]) / \
            float(self.fine*self.coarse_in*self.coarse_out*self.coarse_group)

        weights_bram_usage = bram_memory_resource_model(
                    int(weight_memory_depth),self.weight_t.width) * \
                self.coarse_in*self.coarse_out*self.coarse_group*self.fine

        # bias usage FIXME depth, FIXME bram usage
        bias_memory_depth = float(self.filters) / float(self.coarse_out)
        biases_bram_usage = bram_memory_resource_model(
                    int(bias_memory_depth),self.acc_t.width) * self.coarse_out

        # Total
        return {
            "LUT"  :  sw_rsc['LUT']*self.coarse_in*self.coarse_group +
                      fork_rsc['LUT']*self.coarse_in*self.coarse_group +
                      vector_dot_rsc['LUT']*self.coarse_in*self.coarse_out*self.coarse_group +
                      accum_rsc['LUT']*self.coarse_in*self.coarse_out*self.coarse_group +
                      glue_rsc['LUT']*self.coarse_group +
                      bias_rsc['LUT']*self.coarse_out,
            "FF"   :  sw_rsc['FF']*self.coarse_in*self.coarse_group +
                      fork_rsc['FF']*self.coarse_in*self.coarse_group +
                      vector_dot_rsc['FF']*self.coarse_in*self.coarse_out*self.coarse_group +
                      accum_rsc['FF']*self.coarse_in*self.coarse_out*self.coarse_group +
                      glue_rsc['FF']*self.coarse_group +
                      bias_rsc['FF']*self.coarse_out,
            "BRAM" :  sw_rsc['BRAM']*self.coarse_in*self.coarse_group +
                      fork_rsc['BRAM']*self.coarse_in*self.coarse_group +
                      vector_dot_rsc['BRAM']*self.coarse_in*self.coarse_out*self.coarse_group +
                      accum_rsc['BRAM']*self.coarse_out*self.coarse_group +
                      glue_rsc['BRAM']*self.coarse_group +
                      bias_rsc['BRAM']*self.coarse_out +
                      weights_bram_usage +
                      biases_bram_usage,
            "DSP" :   sw_rsc['DSP']*self.coarse_in*self.coarse_group +
                      fork_rsc['DSP']*self.coarse_in*self.coarse_group +
                      vector_dot_rsc['DSP']*self.coarse_in*self.coarse_out*self.coarse_group +
                      accum_rsc['DSP']*self.coarse_in*self.coarse_out*self.coarse_group +
                      glue_rsc['DSP']*self.coarse_group +
                      bias_rsc['DSP']*self.coarse_out
        }

    def visualise(self, name):

        cluster = pydot.Cluster(name, label=name,
                style="dashed", bgcolor="lightpink")

        # names
        slwin_name = [[""]*self.coarse_in]*self.coarse_group
        fork_name = [[""]*self.coarse_in]*self.coarse_group
        conv_name = [[[""]*self.coarse_in]*self.coarse_out]*self.coarse_group
        accum_name = [[[""]*self.coarse_in]*self.coarse_out]*self.coarse_group
        glue_name = [[""]*self.coarse_out]*self.coarse_group
        bias_name = [[""]*self.coarse_out]*self.coarse_group

        for g in range(self.coarse_group):
            for i in range(self.coarse_in):
                # define names
                slwin_name[g][i] = "_".join([name, "sw", str(g), str(i)])
                fork_name[g][i] = "_".join([name, "fork", str(g), str(i)])
                # add nodes
                cluster.add_node(self.modules["sliding_window"].visualise(slwin_name[g][i]))
                cluster.add_node(self.modules["fork"].visualise(fork_name[g][i]))
                # add edges
                cluster.add_edge(pydot.Edge(slwin_name[g][i], fork_name[g][i]))

                # iterate over coarse out
                for j in range(self.coarse_out):
                    # define names
                    conv_name[g][j][i] = "_".join([name, "conv", str(g), str(j), str(i)])
                    accum_name[g][j][i] = "_".join([name, "accum", str(g), str(j), str(i)])
                    glue_name[g][j] = "_".join([name, "glue", str(g), str(j)])
                    bias_name[g][j] = "_".join([name, "bias", str(g), str(j)])

                    # add nodes
                    cluster.add_node(self.modules["conv"].visualise(conv_name[g][j][i]))
                    cluster.add_node(self.modules["accum"].visualise(accum_name[g][j][i]))

                    # add edges
                    cluster.add_edge(pydot.Edge(fork_name[g][i], conv_name[g][j][i]))
                    cluster.add_edge(pydot.Edge(conv_name[g][j][i], accum_name[g][j][i]))
                    cluster.add_edge(pydot.Edge(accum_name[g][j][i], glue_name[g][j]))

        for g in range(self.coarse_group):
            for j in range(self.coarse_out):

                # add nodes
                cluster.add_node(self.modules["glue"].visualise(glue_name[g][j]))
                cluster.add_node(self.modules["bias"].visualise(bias_name[g][j]))

                # add edges
                cluster.add_edge(pydot.Edge(glue_name[g][j], bias_name[g][j]))


        return cluster, np.array(slwin_name).flatten().tolist(), np.array(bias_name).flatten().tolist()

