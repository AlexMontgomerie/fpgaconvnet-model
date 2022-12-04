import importlib
import math
from typing import Union, List

import pydot
import numpy as np
import torch

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.tools.resource_analytical_model import bram_memory_resource_model
from fpgaconvnet.models.layers import Layer3D

from fpgaconvnet.models.modules import SlidingWindow3D
from fpgaconvnet.models.modules import VectorDot3D
from fpgaconvnet.models.modules import Conv3D
from fpgaconvnet.models.modules import Squeeze3D
from fpgaconvnet.models.modules import Fork3D
from fpgaconvnet.models.modules import Accum3D
from fpgaconvnet.models.modules import Glue3D
from fpgaconvnet.models.modules import Bias3D


class ConvolutionLayer3D(Layer3D):

    def __init__(
            self,
            filters: int,
            rows: int,
            cols: int,
            depth: int,
            channels: int,
            coarse_in: int = 1,
            coarse_out: int = 1,
            coarse_group: int = 1,
            kernel_rows: int = 1,
            kernel_cols: int = 1,
            kernel_depth: int = 1,
            stride_rows: int = 1,
            stride_cols: int = 1,
            stride_depth: int = 1,
            groups: int = 1,
            pad_top: int = 0,
            pad_right: int = 0,
            pad_front: int = 0,
            pad_bottom: int = 0,
            pad_left: int = 0,
            pad_back: int = 0,
            fine: int  = 1,
            input_t: FixedPoint = FixedPoint(16,8),
            output_t: FixedPoint = FixedPoint(16,8),
            weight_t: FixedPoint = FixedPoint(16,8),
            acc_t: FixedPoint = FixedPoint(32,16),
            has_bias: int = 0, # default to no bias for old configs
            backend: str = "chisel", # default to no bias for old configs
        ):

        # initialise parent class
        super().__init__(rows, cols, depth, channels,
                coarse_in, coarse_out)

        # save data types
        self.input_t = input_t
        self.output_t = output_t
        self.weight_t = weight_t
        self.acc_t = acc_t

        # save bias flag
        self.has_bias = has_bias

        # init variables
        self._kernel_rows = kernel_rows
        self._kernel_cols = kernel_cols
        self._kernel_depth = kernel_depth
        self._stride_rows = stride_rows
        self._stride_cols = stride_cols
        self._stride_depth = stride_depth
        self._pad_top = pad_top
        self._pad_right = pad_right
        self._pad_front = pad_front
        self._pad_bottom = pad_bottom
        self._pad_left = pad_left
        self._pad_back = pad_back
        self._groups = groups
        self._coarse_group = coarse_group
        self._fine = fine
        self._filters = filters

        # backend flag
        assert backend in ["hls", "chisel"], f"{backend} is an invalid backend"
        self.backend = backend

        self.modules["sliding_window3d"] = SlidingWindow3D(self.rows_in(), self.cols_in(), self.depth_in(), self.channels_in()//(self.coarse_in*self.coarse_group), self.kernel_rows, self.kernel_cols, self.kernel_depth, self.stride_rows, self.stride_cols, self.stride_depth, self.pad_top, self.pad_right, self.pad_front, self.pad_bottom, self.pad_left, self.pad_back, backend=self.backend)

        if self.backend == "hls":

            self.modules["fork3d"] = Fork3D(self.rows_out(), self.cols_out(), self.depth_out(), self.channels_in()//(self.coarse_in*self.coarse_group), self.kernel_rows, self.kernel_cols, self.kernel_depth, self.coarse_out, backend=self.backend)

            self.modules["conv3d"] = Conv3D(self.rows_out(), self.cols_out(), self.depth_out(), self.channels_in()//(self.coarse_in*self.coarse_group), self.filters//(self.coarse_out*self.groups), self.kernel_rows, self.kernel_cols, self.kernel_depth, backend=self.backend)

            self.modules["accum3d"] = Accum3D(self.rows_out(), self.cols_out(), self.depth_out(), self.channels_in()//(self.coarse_in*self.groups), self.filters//(self.coarse_out*self.groups), 1, backend=self.backend)

        elif self.backend == "chisel":

            self.modules["squeeze3d"] = Squeeze3D(self.rows_out(), self.cols_out(), self.depth_out(), self.channels_in()//(self.coarse_in*self.coarse_group), self.kernel_rows*self.kernel_cols*self.kernel_depth, self.fine, backend=self.backend)

            self.modules["fork3d"] = Fork3D(self.rows_out(), self.cols_out(), self.depth_out(), self.channels_in()//(self.coarse_in*self.coarse_group), self.fine, 1, 1, self.coarse_out, backend=self.backend)

            self.modules["vector_dot3d"] = VectorDot3D(self.rows_out(), self.cols_out(), self.depth_out(), self.channels_in()//(self.coarse_in*self.coarse_group), self.filters//(self.coarse_out*self.groups), self.fine, backend=self.backend)

            self.modules["accum3d"] = Accum3D(self.rows_out(), self.cols_out(), self.depth_out(), (self.kernel_rows*self.kernel_cols*self.kernel_depth*self.channels_in())//(self.fine*self.coarse_in*self.groups), self.filters//(self.coarse_out*self.groups), 1, backend=self.backend)

        self.modules["glue3d"] = Glue3D(self.rows_out(), self.cols_out(), self.depth_out(), 1, int(self.filters/self.coarse_out), self.coarse_in, self.coarse_out, backend=self.backend) # TODO

        self.modules["bias3d"] = Bias3D(self.rows_out(), self.cols_out(), self.depth_out(), 1, self.filters, backend=self.backend) # TODO

        # update modules
        self.update()

    @property
    def kernel_rows(self) -> int:
        return self._kernel_rows

    @property
    def kernel_cols(self) -> int:
        return self._kernel_cols

    @property
    def kernel_depth(self) -> int:
        return self._kernel_depth

    @property
    def stride_rows(self) -> int:
        return self._stride_rows

    @property
    def stride_cols(self) -> int:
        return self._stride_cols

    @property
    def stride_depth(self) -> int:
        return self._stride_depth

    @property
    def pad_top(self) -> int:
        return self._pad_top

    @property
    def pad_right(self) -> int:
        return self._pad_right

    @property
    def pad_front(self) -> int:
        return self._pad_front

    @property
    def pad_bottom(self) -> int:
        return self._pad_bottom

    @property
    def pad_left(self) -> int:
        return self._pad_left

    @property
    def pad_back(self) -> int:
        return self._pad_back

    @property
    def groups(self) -> int:
        return self._groups

    @property
    def coarse_group(self) -> int:
        return self._coarse_group

    @property
    def fine(self) -> int:
        return self._fine

    @property
    def filters(self) -> int:
        return self._filters

    @kernel_rows.setter
    def kernel_rows(self, val: int) -> None:
        self._kernel_rows = val
        self.update()

    @kernel_cols.setter
    def kernel_cols(self, val: int) -> None:
        self._kernel_cols = val
        self.update()

    @kernel_depth.setter
    def kernel_depth(self, val: int) -> None:
        self._kernel_depth = val
        self.update()

    @stride_rows.setter
    def stride_rows(self, val: int) -> None:
        self._stride_rows = val
        self.update()

    @stride_cols.setter
    def stride_cols(self, val: int) -> None:
        self._stride_cols = val
        self.update()

    @stride_depth.setter
    def stride_depth(self, val: int) -> None:
        self._stride_depth = val
        self.update()

    @pad_top.setter
    def pad_top(self, val: int) -> None:
        self._pad_top = val
        self.update()

    @pad_right.setter
    def pad_right(self, val: int) -> None:
        self._pad_right = val
        self.update()

    @pad_front.setter
    def pad_front(self, val: int) -> None:
        self._pad_front = val
        self.update()

    @pad_bottom.setter
    def pad_bottom(self, val: int) -> None:
        self._pad_bottom = val
        self.update()

    @pad_left.setter
    def pad_left(self, val: int) -> None:
        self._pad_left = val
        self.update()

    @pad_back.setter
    def pad_back(self, val: int) -> None:
        self._pad_back = val
        self.update()

    @groups.setter
    def groups(self, val: int) -> None:
        self._groups = val
        self.update()

    @fine.setter
    def fine(self, val: int) -> None:
        self._fine = val
        self.update()

    @filters.setter
    def filters(self, val: int) -> None:
        self._filters = val
        self.update()

    @coarse_group.setter
    def coarse_group(self, val: int) -> None:
        assert(val in self.get_coarse_group_feasible())
        self._coarse_group = val
        self.update()

    def rows_out(self) -> int:
        return self.modules["sliding_window3d"].rows_out()

    def cols_out(self) -> int:
        return self.modules["sliding_window3d"].cols_out()

    def depth_out(self) -> int:
        return self.modules["sliding_window3d"].depth_out()

    def channels_out(self) -> int:
        return self.filters

    def streams_in(self) -> int:
        """
        Returns
        -------
        int
            number of parallel streams into the layer.
        """
        return self.coarse_in*self.coarse_group

    def streams_out(self) -> int:
        """
        Returns
        -------
        int
            number of parallel streams out of the layer.
        """
        return self.coarse_out*self.coarse_group


    def update(self):

        # sliding window
        self.modules['sliding_window3d'].rows     = self.rows
        self.modules['sliding_window3d'].cols     = self.cols
        self.modules['sliding_window3d'].depth    = self.depth
        self.modules['sliding_window3d'].channels = self.channels//(self.coarse_in*self.coarse_group)
        self.modules['sliding_window3d'].data_width   = self.input_t.width

        if self.backend == "chisel":
            # squeeze3d
            self.modules['squeeze3d'].rows     = self.rows_out()
            self.modules['squeeze3d'].cols     = self.cols_out()
            self.modules['squeeze3d'].depth    = self.depth_out()
            self.modules['squeeze3d'].channels = self.channels//(self.coarse_in*self.coarse_group)
            self.modules['squeeze3d'].coarse_out = self.fine
            self.modules['squeeze3d'].data_width = self.input_t.width

        # fork3d
        self.modules['fork3d'].rows     = self.rows_out()
        self.modules['fork3d'].cols     = self.cols_out()
        self.modules['fork3d'].depth    = self.depth_out()
        self.modules['fork3d'].channels = self.channels_in()//(self.coarse_in*self.coarse_group)
        self.modules['fork3d'].coarse   = self.coarse_out
        self.modules['fork3d'].data_width     = self.input_t.width
        if self.backend == "chisel":
            self.modules['fork3d'].kernel_rows = self.fine
            self.modules['fork3d'].kernel_cols = 1
            self.modules['fork3d'].kernel_depth = 1

        if self.backend == "hls":
            # TODO: check the group parameter
            self.modules['conv3d'].rows     = self.rows_out()
            self.modules['conv3d'].cols     = self.cols_out()
            self.modules['conv3d'].depth    = self.depth_out()
            self.modules['conv3d'].channels = self.channels_in()//(self.coarse_in*self.coarse_group)
            self.modules['conv3d'].filters  = self.filters//(self.coarse_out*self.coarse_group)
            self.modules['conv3d'].fine     = self.fine
            self.modules['conv3d'].data_width     = self.input_t.width
            self.modules['conv3d'].weight_width   = self.weight_t.width
            self.modules['conv3d'].acc_width      = self.acc_t.width
        elif self.backend == "chisel":
            # kernel dot
            self.modules['vector_dot3d'].rows     = self.rows_out()
            self.modules['vector_dot3d'].cols     = self.cols_out()
            self.modules['vector_dot3d'].depth    = self.depth_out()
            self.modules['vector_dot3d'].channels = (
                    self.channels*self.kernel_rows*self.kernel_cols*self.kernel_depth)//(
                    self.fine*self.coarse_in*self.groups)
            self.modules['vector_dot3d'].filters  = self.filters//(self.coarse_out*self.coarse_group)
            self.modules['vector_dot3d'].fine     = self.fine
            self.modules['vector_dot3d'].data_width     = self.input_t.width
            self.modules['vector_dot3d'].weight_width   = self.weight_t.width
            self.modules['vector_dot3d'].acc_width      = self.acc_t.width

        # accum3d
        self.modules['accum3d'].rows     = self.rows_out()
        self.modules['accum3d'].cols     = self.cols_out()
        self.modules['accum3d'].depth    = self.depth_out()
        self.modules['accum3d'].filters  = self.filters//(self.coarse_out*self.coarse_group)
        self.modules['accum3d'].data_width    = self.acc_t.width
        if self.backend == "hls":
            # TODO: check the group parameter
            self.modules['accum3d'].channels  = self.channels_in()//(self.coarse_in*self.coarse_group)
        elif self.backend == "chisel":
            self.modules['accum3d'].channels = (
                    self.channels*self.kernel_rows*self.kernel_cols*self.kernel_depth)//(
                    self.fine*self.coarse_in*self.coarse_group)
            self.modules['accum3d'].groups   = 1

        # glue3d
        self.modules['glue3d'].rows       = self.rows_out()
        self.modules['glue3d'].cols       = self.cols_out()
        self.modules['glue3d'].depth      = self.depth_out()
        self.modules['glue3d'].filters    = self.filters//self.coarse_group
        self.modules['glue3d'].coarse_in  = self.coarse_in
        self.modules['glue3d'].coarse_out = self.coarse_out
        self.modules['glue3d'].data_width = self.acc_t.width

        # bias3d
        self.modules['bias3d'].rows           = self.rows_out()
        self.modules['bias3d'].cols           = self.cols_out()
        self.modules['bias3d'].depth          = self.depth_out()
        self.modules['bias3d'].filters        = self.filters
        self.modules['bias3d'].data_width     = self.output_t.width
        self.modules['bias3d'].biases_width   = self.acc_t.width

    def layer_info(self,parameters,batch_size=1):
        Layer3D.layer_info(self, parameters, batch_size)
        parameters.filters      = self.filters
        parameters.groups       = self.groups
        parameters.coarse_group = self.coarse_group
        parameters.fine         = self.fine
        parameters.kernel_rows  = self.kernel_rows
        parameters.kernel_cols  = self.kernel_cols
        parameters.kernel_depth = self.kernel_depth
        parameters.stride_rows  = self.stride_rows
        parameters.stride_cols  = self.stride_cols
        parameters.stride_depth = self.stride_depth
        parameters.pad_top      = self.pad_top
        parameters.pad_right    = self.pad_right
        parameters.pad_front    = self.pad_front
        parameters.pad_bottom   = self.pad_bottom
        parameters.pad_left     = self.pad_left
        parameters.pad_back     = self.pad_back
        parameters.has_bias     = self.has_bias
        self.input_t.to_protobuf(parameters.input_t)
        self.output_t.to_protobuf(parameters.output_t)
        self.weight_t.to_protobuf(parameters.weight_t)
        self.acc_t.to_protobuf(parameters.acc_t)
        parameters.data_t.Clear()

    def get_coarse_group_feasible(self):
        return get_factors(self.groups)

    def get_coarse_in_feasible(self):
        return get_factors(int(self.channels_in())//self.groups)

    def get_coarse_out_feasible(self):
        return get_factors(int(self.channels_out())//self.groups)

    def get_fine_feasible(self):
        if self.backend == "chisel":
            return get_factors(self.kernel_rows*self.kernel_cols*self.kernel_depth)
        elif self.backend == "hls":
            if self.kernel_depth != self.kernel_rows and self.kernel_rows == self.kernel_cols:
                if self.kernel_depth == 1:
                    fine_feasible = [1, self.kernel_rows, self.kernel_rows * self.kernel_cols]
                elif self.kernel_rows == 1:
                    fine_feasible = [1, self.kernel_depth]
                else:
                    fine_feasible = [
                        1,
                        self.kernel_depth,
                        self.kernel_rows,
                        self.kernel_depth * self.kernel_rows,
                        self.kernel_rows * self.kernel_cols,
                        self.kernel_depth * self.kernel_rows * self.kernel_cols,
                    ]
            elif self.kernel_depth == self.kernel_rows and self.kernel_rows == self.kernel_cols:
                if self.kernel_depth == 1:
                    fine_feasible = [1]
                else:
                    fine_feasible = [
                        1,
                        self.kernel_depth,
                        self.kernel_depth * self.kernel_rows,
                        self.kernel_depth * self.kernel_rows * self.kernel_cols,
                    ]
            else:
                fine_feasible = [
                    1,
                    self.kernel_depth,
                    self.kernel_rows,
                    self.kernel_cols,
                    self.kernel_depth * self.kernel_rows,
                    self.kernel_depth * self.kernel_cols,
                    self.kernel_rows * self.kernel_cols,
                    self.kernel_depth * self.kernel_rows * self.kernel_cols,
                ]
            return fine_feasible

    def get_weights_reloading_feasible(self):
        return get_factors(self.filters//(self.groups*self.coarse_out))

    def get_parameters_size(self):
        weights_size = self.channels_in() * ( self.filters // self.groups ) * self.kernel_rows * self.kernel_cols * self.kernel_depth
        bias_size = 0
        return {
            "weights"   : weights_size,
            "bias"      : bias_size
        }

    def get_operations(self):
        return self.kernel_rows*self.kernel_cols*self.kernel_depth*self.channels_in()*self.filters*self.rows_out()*self.cols_out()*self.depth_out()

    def resource(self):

        if self.backend == "chisel":

            # get module resource models
            sw_rsc          = self.modules['sliding_window3d'].rsc()
            squeeze_rsc     = self.modules['squeeze3d'].rsc()
            fork_rsc        = self.modules['fork3d'].rsc()
            vector_dot_rsc  = self.modules['vector_dot3d'].rsc()
            accum_rsc       = self.modules['accum3d'].rsc()
            glue_rsc        = self.modules['glue3d'].rsc()
            bias_rsc        = self.modules['bias3d'].rsc()

            # remove redundant modules
            if self.kernel_rows == 1 and self.kernel_cols == 1 and self.kernel_depth == 1:
                sw_rsc      = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            if self.fine == self.kernel_rows*self.kernel_cols*self.kernel_depth:
                squeeze_rsc = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            if self.coarse_out == 1:
                fork_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            if int(self.channels_in()/(self.coarse_in*self.coarse_group)) == 1:
                accum_rsc   = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            if self.coarse_in == 1:
                glue_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            if self.has_bias:
                bias_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}

            # accumulate resource usage based on coarse factors
            rsc = { rsc_type: (
                sw_rsc[rsc_type]*self.coarse_in*self.coarse_group +
                squeeze_rsc[rsc_type]*self.coarse_in*self.coarse_group +
                fork_rsc[rsc_type]*self.coarse_in*self.coarse_group +
                vector_dot_rsc[rsc_type]*self.coarse_in*self.coarse_out*self.coarse_group +
                accum_rsc[rsc_type]*self.coarse_in*self.coarse_out*self.coarse_group +
                glue_rsc[rsc_type]*self.coarse_out*self.coarse_group +
                bias_rsc[rsc_type]*self.coarse_out
            ) for rsc_type in ["LUT", "FF", "DSP", "BRAM"] }

        elif self.backend == "hls":

            # get module resource models
            sw_rsc          = self.modules['sliding_window3d'].rsc()
            fork_rsc        = self.modules['fork3d'].rsc()
            conv_rsc        = self.modules['conv3d'].rsc()
            accum_rsc       = self.modules['accum3d'].rsc()
            glue_rsc        = self.modules['glue3d'].rsc()
            bias_rsc        = self.modules['bias3d'].rsc()

            # remove redundant modules
            if self.kernel_rows == 1 and self.kernel_cols == 1 and self.kernel_depth == 1:
                sw_rsc      = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            if self.coarse_out == 1:
                fork_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            if int(self.channels_in()/(self.coarse_in*self.coarse_group)) == 1:
                accum_rsc   = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            if self.coarse_in == 1:
                glue_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            if self.has_bias:
                bias_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}

            # accumulate resource usage based on coarse factors
            rsc = { rsc_type: (
                sw_rsc[rsc_type]*self.coarse_in*self.coarse_group +
                fork_rsc[rsc_type]*self.coarse_in*self.coarse_group +
                conv_rsc[rsc_type]*self.coarse_in*self.coarse_out*self.coarse_group +
                accum_rsc[rsc_type]*self.coarse_in*self.coarse_out*self.coarse_group +
                glue_rsc[rsc_type]*self.coarse_out*self.coarse_group +
                bias_rsc[rsc_type]*self.coarse_out
            ) for rsc_type in ["LUT", "FF", "DSP", "BRAM"] }

        # weight usage
        weight_memory_depth = float((self.filters/self.groups)* \
                                    self.channels_in()* \
                                    self.kernel_rows* \
                                    self.kernel_cols* \
                                    self.kernel_depth) / \
            float(self.fine*self.coarse_in*self.coarse_out*self.coarse_group)

        weights_bram_usage = bram_memory_resource_model(
                    int(weight_memory_depth), self.weight_t.width*self.fine) * \
                self.coarse_in*self.coarse_out*self.coarse_group

        # bias usage FIXME depth, FIXME bram usage
        bias_memory_depth = float(self.filters) / float(self.coarse_out)
        biases_bram_usage = bram_memory_resource_model(
                    int(bias_memory_depth),self.acc_t.width) * self.coarse_out

        # add weights and bias to resources
        rsc["BRAM"] += weights_bram_usage + biases_bram_usage

        # return total resource
        return rsc

    def visualise(self, name):
        cluster = pydot.Cluster(name, label=name,
                style="dashed", bgcolor="lightpink")

        # names
        slwin_name = [[""]*self.coarse_in]*self.coarse_group
        fork_name = [[""]*self.coarse_in]*self.coarse_group
        # conv_name = [[[""]*self.coarse_in]*self.coarse_out]*self.coarse_group
        vector_dot_name = [[[""]*self.coarse_in]*self.coarse_out]*self.coarse_group
        accum_name = [[[""]*self.coarse_in]*self.coarse_out]*self.coarse_group
        glue_name = [[""]*self.coarse_out]*self.coarse_group
        bias_name = [[""]*self.coarse_out]*self.coarse_group

        for g in range(self.coarse_group):
            for i in range(self.coarse_in):
                # define names
                slwin_name[g][i] = "_".join([name, "sw3d", str(g), str(i)])
                fork_name[g][i] = "_".join([name, "fork3d", str(g), str(i)])
                # add nodes
                cluster.add_node(self.modules["sliding_window3d"].visualise(slwin_name[g][i]))
                cluster.add_node(self.modules["fork3d"].visualise(fork_name[g][i]))
                # add edges
                cluster.add_edge(pydot.Edge(slwin_name[g][i], fork_name[g][i]))

                # iterate over coarse out
                for j in range(self.coarse_out):
                    # define names
                    # conv_name[g][j][i] = "_".join([name, "conv3d", str(g), str(j), str(i)])
                    vector_dot_name[g][j][i] = "_".join([name, "vector_dot3d", str(g), str(j), str(i)])
                    accum_name[g][j][i] = "_".join([name, "accum3d", str(g), str(j), str(i)])
                    glue_name[g][j] = "_".join([name, "glue3d", str(g), str(j)])
                    bias_name[g][j] = "_".join([name, "bias3d", str(g), str(j)])

                    # add nodes
                    # cluster.add_node(self.modules["conv3d"].visualise(conv_name[g][j][i]))
                    cluster.add_node(self.modules["vector_dot3d"].visualise(vector_dot_name[g][j][i]))
                    cluster.add_node(self.modules["accum3d"].visualise(accum_name[g][j][i]))

                    # add edges
                    # cluster.add_edge(pydot.Edge(fork_name[g][i], conv_name[g][j][i]))
                    cluster.add_edge(pydot.Edge(fork_name[g][i], vector_dot_name[g][j][i]))
                    # cluster.add_edge(pydot.Edge(conv_name[g][j][i], accum_name[g][j][i]))
                    cluster.add_edge(pydot.Edge(vector_dot_name[g][j][i], accum_name[g][j][i]))
                    cluster.add_edge(pydot.Edge(accum_name[g][j][i], glue_name[g][j]))

        for g in range(self.coarse_group):
            for j in range(self.coarse_out):

                # add nodes
                cluster.add_node(self.modules["glue3d"].visualise(glue_name[g][j]))
                cluster.add_node(self.modules["bias3d"].visualise(bias_name[g][j]))

                # add edges
                cluster.add_edge(pydot.Edge(glue_name[g][j], bias_name[g][j]))


        return cluster, np.array(slwin_name).flatten().tolist(), np.array(bias_name).flatten().tolist()

    def functional_model(self,data,weights,bias,batch_size=1):

        assert data.shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.depth_in()   , "ERROR (data): invalid depth dimension"
        assert data.shape[3] == self.channels_in(), "ERROR (data): invalid channel dimension"

        assert weights.shape[0] == self.filters ,   "ERROR (weights): invalid filter dimension"
        assert weights.shape[1] == self.channels//self.groups,\
                                                    "ERROR (weights): invalid channel dimension"
        assert weights.shape[2] == self.kernel_rows,\
                                                    "ERROR (weights): invalid kernel dimension"
        assert weights.shape[3] == self.kernel_cols,\
                                                    "ERROR (weights): invalid kernel dimension"
        assert weights.shape[4] == self.kernel_depth,\
                                                    "ERROR (weights): invalid kernel dimension"

        assert bias.shape[0] == self.filters  ,     "ERROR (bias): invalid filter dimension"

        # instantiate convolution layer
        # convolution_layer = torch.nn.Conv3d(self.channels_in(), self.filters, (self.kernel_depth, self.kernel_rows, self.kernel_cols), stride=(self.stride_depth, self.stride_rows, self.stride_cols), padding=(self.pad_front, self.pad_top, self.pad_right), groups=self.groups, bias=True)
        convolution_layer = torch.nn.Conv3d(self.channels_in(), self.filters, (self.kernel_depth, self.kernel_rows, self.kernel_cols), stride=(self.stride_depth, self.stride_rows, self.stride_cols), padding=0, groups=self.groups, bias=True)

        # update weights
        convolution_layer.weight = torch.nn.Parameter(torch.from_numpy(weights))

        # update bias
        convolution_layer.bias = torch.nn.Parameter(torch.from_numpy(bias))

        # get the padding
        padding = [
            self.pad_left,
            self.pad_right,
            self.pad_top,
            self.pad_bottom,
            self.pad_front,
            self.pad_back,
        ]

        # return output featuremap
        data = np.moveaxis(data, [-1, -2], [0, 1])
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        data = torch.nn.functional.pad(torch.from_numpy(data), padding, "constant", 0.0)
        data = convolution_layer(data).detach().numpy()
        print(data.shape)
        return data
        # return convolution_layer(data).detach().numpy()

