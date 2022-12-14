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
from fpgaconvnet.models.layers import Layer

from fpgaconvnet.models.modules import SlidingWindow
from fpgaconvnet.models.modules import VectorDot
from fpgaconvnet.models.modules import Conv
from fpgaconvnet.models.modules import Squeeze
from fpgaconvnet.models.modules import Fork
from fpgaconvnet.models.modules import Accum
from fpgaconvnet.models.modules import Glue
from fpgaconvnet.models.modules import Bias


class ConvolutionLayer(Layer):

    def __init__(
            self,
            filters: int,
            rows: int,
            cols: int,
            channels: int,
            coarse_in: int = 1,
            coarse_out: int = 1,
            coarse_group: int = 1,
            kernel_rows: int = 1,
            kernel_cols: int = 1,
            stride_rows: int = 2,
            stride_cols: int = 2,
            groups: int = 1,
            pad_top: int = 0,
            pad_right: int = 0,
            pad_bottom: int = 0,
            pad_left: int = 0,
            fine: int  = 1,
            input_t: FixedPoint = FixedPoint(16,8),
            output_t: FixedPoint = FixedPoint(16,8),
            weight_t: FixedPoint = FixedPoint(16,8),
            acc_t: FixedPoint = FixedPoint(32,16),
            has_bias: int = 0, # default to no bias for old configs
            backend: str = "chisel", # default to no bias for old configs
            double_buffered: bool = True,
            stream_weights: bool = True,
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
        self._kernel_rows = kernel_rows
        self._kernel_cols = kernel_cols
        self._stride_rows = stride_rows
        self._stride_cols = stride_cols
        self._pad_top       = pad_top
        self._pad_right     = pad_right
        self._pad_bottom    = pad_bottom
        self._pad_left      = pad_left
        self._groups = groups
        self._coarse_group = coarse_group
        self._fine = fine
        self._filters = filters

        # weights buffering flag
        self.double_buffered = double_buffered
        self.stream_weights = stream_weights

        # backend flag
        assert backend in ["hls", "chisel"], f"{backend} is an invalid backend"
        self.backend = backend

        self.modules["sliding_window"] = SlidingWindow(self.rows_in(), self.cols_in(),
                self.channels_in()//(self.coarse_in*self.coarse_group), self.kernel_size,
                self.stride, self.pad_top, self.pad_right, self.pad_bottom, self.pad_left,
                backend=self.backend)

        if self.backend == "hls":

            self.modules["fork"] = Fork(self.rows_out(), self.cols_out(),
                    self.channels_in()//(self.coarse_in*self.coarse_group),
                    self.kernel_size, self.coarse_out, backend=self.backend)

            self.modules["Conv"] = Conv(self.rows_out(), self.cols_out(),
                    self.channels_in()//(self.coarse_in*self.coarse_group),
                    self.filters//(self.coarse_out*self.groups), self.kernel_size,
                    backend=self.backend)

            self.modules["accum"] = Accum(self.rows_out(), self.cols_out(),
                    self.channels_in()//(self.coarse_in*self.groups),
                    self.filters//(self.coarse_out*self.groups), 1,
                    backend=self.backend)

        elif self.backend == "chisel":

            self.modules["squeeze"] = Squeeze(self.rows_out(), self.cols_out(),
                    self.channels_in()//(self.coarse_in*self.coarse_group),
                    self.kernel_size[0]*self.kernel_size[1], self.fine,
                    backend=self.backend)

            self.modules["fork"] = Fork(self.rows_out(), self.cols_out(),
                    self.channels_in()//(self.coarse_in*self.coarse_group),
                    [self.fine, 1], self.coarse_out, backend=self.backend)

            self.modules["vector_dot"] = VectorDot(self.rows_out(), self.cols_out(),
                    self.channels_in()//(self.coarse_in*self.coarse_group),
                    self.filters//(self.coarse_out*self.groups), self.fine,
                    backend=self.backend)

            self.modules["accum"] = Accum(self.rows_out(), self.cols_out(),
                    (self.kernel_size[0]*self.kernel_size[1]*self.channels_in())//(
                        self.fine*self.coarse_in*self.groups),
                    self.filters//(self.coarse_out*self.groups), 1,
                    backend=self.backend)

        self.modules["glue"] = Glue(self.rows_out(), self.cols_out(), 1,
                int(self.filters/self.coarse_out), self.coarse_in, self.coarse_out,
                backend=self.backend) # TODO

        self.modules["bias"] = Bias(self.rows_out(), self.cols_out(), 1, self.filters,
                backend=self.backend) # TODO

        # update modules
        self.update()

    @property
    def kernel_size(self) -> List[int]:
        return [ self._kernel_rows, self._kernel_cols ]

    @property
    def kernel_rows(self) -> int:
        return self._kernel_rows

    @property
    def kernel_cols(self) -> int:
        return self._kernel_cols

    @property
    def stride(self) -> List[int]:
        return [ self._stride_rows, self._stride_cols ]

    @property
    def stride_rows(self) -> int:
        return self._stride_rows

    @property
    def stride_cols(self) -> int:
        return self._stride_cols

    @property
    def pad(self) -> List[int]:
        return [
            self._pad_top,
            self._pad_left,
            self._pad_bottom,
            self._pad_right,
        ]

    @property
    def pad_top(self) -> int:
        return self._pad_top

    @property
    def pad_right(self) -> int:
        return self._pad_right

    @property
    def pad_bottom(self) -> int:
        return self._pad_bottom

    @property
    def pad_left(self) -> int:
        return self._pad_left

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

    @kernel_size.setter
    def kernel_size(self, val: List[int]) -> None:
        self._kernel_rows = val[0]
        self._kernel_cols = val[1]
        # self.update()

    @kernel_rows.setter
    def kernel_rows(self, val: int) -> None:
        self._kernel_rows = val
        # self.update()

    @kernel_cols.setter
    def kernel_cols(self, val: int) -> None:
        self._kernel_cols = val
        # self.update()

    @stride.setter
    def stride(self, val: List[int]) -> None:
        self._stride_rows = val[0]
        self._stride_cols = val[1]
        # self.update()

    @stride_rows.setter
    def stride_rows(self, val: int) -> None:
        self._stride_rows = val
        # self.update()

    @stride_cols.setter
    def stride_cols(self, val: int) -> None:
        self._stride_cols = val
        # self.update()

    @pad.setter
    def pad(self, val: List[int]) -> None:
        self._pad_top    = val[0]
        self._pad_right  = val[3]
        self._pad_bottom = val[2]
        self._pad_left   = val[1]
        # self.update()

    @pad_top.setter
    def pad_top(self, val: int) -> None:
        self._pad_top = val
        # self.update()

    @pad_right.setter
    def pad_right(self, val: int) -> None:
        self._pad_right = val
        # self.update()

    @pad_bottom.setter
    def pad_bottom(self, val: int) -> None:
        self._pad_bottom = val
        # self.update()

    @pad_left.setter
    def pad_left(self, val: int) -> None:
        self._pad_left = val
        # self.update()

    @groups.setter
    def groups(self, val: int) -> None:
        self._groups = val
        # self.update()

    @fine.setter
    def fine(self, val: int) -> None:
        self._fine = val
        # self.update()

    @filters.setter
    def filters(self, val: int) -> None:
        self._filters = val
        # self.update()

    @coarse_group.setter
    def coarse_group(self, val: int) -> None:
        assert(val in self.get_coarse_group_feasible())
        self._coarse_group = val
        # self.update()

    def rows_out(self) -> int:
        return self.modules["sliding_window"].rows_out()

    def cols_out(self) -> int:
        return self.modules["sliding_window"].cols_out()

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
        self.modules['sliding_window'].rows     = self.rows
        self.modules['sliding_window'].cols     = self.cols
        self.modules['sliding_window'].channels = self.channels//(self.coarse_in*self.coarse_group)
        self.modules['sliding_window'].data_width   = self.input_t.width

        if self.backend == "chisel":
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
        if self.backend == "chisel":
            self.modules['fork'].kernel_size = [self.fine, 1]

        if self.backend == "hls":
            # TODO: check the group parameter
            self.modules['conv'].rows     = self.rows_out()
            self.modules['conv'].cols     = self.cols_out()
            self.modules['conv'].channels = self.channels_in()//(self.coarse_in*self.coarse_group)
            self.modules['conv'].filters  = self.filters//(self.coarse_out*self.coarse_group)
            self.modules['conv'].fine     = self.fine
            self.modules['conv'].data_width     = self.input_t.width
            self.modules['conv'].weight_width   = self.weight_t.width
            self.modules['conv'].acc_width      = self.acc_t.width
        elif self.backend == "chisel":
            # kernel dot
            self.modules['vector_dot'].rows     = self.rows_out()
            self.modules['vector_dot'].cols     = self.cols_out()
            self.modules['vector_dot'].channels = (
                    self.channels*self.kernel_size[0]*self.kernel_size[1])//(
                    self.fine*self.coarse_in*self.groups)
            self.modules['vector_dot'].filters  = self.filters//(self.coarse_out*self.coarse_group)
            self.modules['vector_dot'].fine     = self.fine
            self.modules['vector_dot'].data_width     = self.input_t.width
            self.modules['vector_dot'].weight_width   = self.weight_t.width
            self.modules['vector_dot'].acc_width      = self.acc_t.width

        # accum
        self.modules['accum'].rows     = self.rows_out()
        self.modules['accum'].cols     = self.cols_out()
        self.modules['accum'].filters  = self.filters//(self.coarse_out*self.coarse_group)
        self.modules['accum'].data_width    = self.acc_t.width
        if self.backend == "hls":
            # TODO: check the group parameter
            self.modules['accum3d'].channels  = self.channels_in()//(self.coarse_in*self.coarse_group)
        elif self.backend == "chisel":
            self.modules['accum'].channels = (
                    self.channels*self.kernel_size[0]*self.kernel_size[1])//(
                    self.fine*self.coarse_in*self.coarse_group)
            self.modules['accum'].groups   = 1

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

    def layer_info(self,parameters,batch_size=1):
        Layer.layer_info(self, parameters, batch_size)
        parameters.filters      = self.filters
        parameters.groups       = self.groups
        parameters.coarse_group = self.coarse_group
        parameters.fine         = self.fine
        parameters.kernel_size.extend(self.kernel_size)
        parameters.kernel_rows  = self.kernel_rows
        parameters.kernel_cols  = self.kernel_cols
        parameters.stride.extend(self.stride)
        parameters.stride_rows  = self.stride_rows
        parameters.stride_cols  = self.stride_cols
        parameters.pad_top      = self.pad_top
        parameters.pad_right    = self.pad_right
        parameters.pad_bottom   = self.pad_bottom
        parameters.pad_left     = self.pad_left
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
            return get_factors(self.kernel_size[0]*self.kernel_size[1])
        elif self.backend == "hls":
            if self.kernel_size[0] != self.kernel_size[1]:
                # assert(self.kernel_size[0] == 1 or self.kernel_size[1] == 1)
                return [ 1, max(self.kernel_size[0],self.kernel_size[1])]
            else:
                return [ 1, self.kernel_size[0], self.kernel_size[0]*self.kernel_size[1] ]

    def get_weights_reloading_feasible(self):
        return get_factors(self.filters//(self.groups*self.coarse_out))

    def get_parameters_size(self):
        weights_size = self.channels_in() * ( self.filters // self.groups ) * self.kernel_size[0] * self.kernel_size[1]
        bias_size = 0
        return {
            "weights"   : weights_size,
            "bias"      : bias_size
        }

    def get_operations(self):
        return self.kernel_size[0]*self.kernel_size[1]*self.channels_in()*self.filters*self.rows_out()*self.cols_out()

    def resource(self):

        if self.backend == "chisel":

            # get module resource models
            sw_rsc          = self.modules['sliding_window'].rsc()
            squeeze_rsc     = self.modules['squeeze'].rsc()
            fork_rsc        = self.modules['fork'].rsc()
            vector_dot_rsc  = self.modules['vector_dot'].rsc()
            accum_rsc       = self.modules['accum'].rsc()
            glue_rsc        = self.modules['glue'].rsc()
            bias_rsc        = self.modules['bias'].rsc()

            # remove redundant modules
            if self.kernel_size[0] == 1 and self.kernel_size[1] == 1:
                sw_rsc      = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            if self.fine == self.kernel_size[0]*self.kernel_size[1]:
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

        # weight usage
        weight_memory_depth = float((self.filters/self.groups)* \
                                    self.channels_in()* \
                                    self.kernel_size[0]* \
                                    self.kernel_size[1]) / \
            float(self.fine*self.coarse_in*self.coarse_out*self.coarse_group)

        if self.double_buffered:
            weight_memory_depth *= 2

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
        pass
        """
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
        """

    def functional_model(self,data,weights,bias,batch_size=1):

        assert data.shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels_in(), "ERROR (data): invalid channel dimension"

        assert weights.shape[0] == self.filters ,   "ERROR (weights): invalid filter dimension"
        assert weights.shape[1] == self.channels//self.groups,\
                                                    "ERROR (weights): invalid channel dimension"
        assert weights.shape[2] == self.kernel_size[0],\
                                                    "ERROR (weights): invalid kernel dimension"
        assert weights.shape[3] == self.kernel_size[1],\
                                                    "ERROR (weights): invalid kernel dimension"

        assert bias.shape[0] == self.filters  ,     "ERROR (bias): invalid filter dimension"

        # instantiate convolution layer
        convolution_layer = torch.nn.Conv2d(self.channels_in(), self.filters, self.kernel_size,
                stride=self.stride, padding=0, groups=self.groups)

        # update weights
        convolution_layer.weight = torch.nn.Parameter(torch.from_numpy(weights))

        # update bias
        convolution_layer.bias = torch.nn.Parameter(torch.from_numpy(bias))

        # get the padding
        padding = [
            self.pad_left,
            self.pad_right,
            self.pad_top,
            self.pad_bottom
        ]

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        data = torch.nn.functional.pad(torch.from_numpy(data), padding, "constant", 0.0)
        data = convolution_layer(data).detach().numpy()
        print(data.shape)
        return data
        # return convolution_layer(data).detach().numpy()

