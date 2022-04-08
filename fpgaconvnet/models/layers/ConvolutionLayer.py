import numpy as np
import math
import pydot
import torch
from typing import Union, List

from fpgaconvnet.models.layers.utils import get_factors

from fpgaconvnet.tools.resource_model import bram_memory_resource_model

from fpgaconvnet.models.modules import SlidingWindow
from fpgaconvnet.models.modules import Conv
from fpgaconvnet.models.modules import Fork
from fpgaconvnet.models.modules import Accum
from fpgaconvnet.models.modules import Glue
from fpgaconvnet.models.modules import Bias
from fpgaconvnet.models.layers import Layer

class ConvolutionLayer(Layer):

    def format_kernel_size(self, kernel_size):
        if isinstance(kernel_size, int):
            return [kernel_size, kernel_size]
        elif isinstance(kernel_size, list):
            assert len(kernel_size) == 2, "Must specify two kernel dimensions"
            return kernel_size
        else:
            raise TypeError

    def format_stride(self, stride):
        if isinstance(stride, int):
            return [stride, stride]
        elif isinstance(stride, list):
            assert len(stride) == 2, "Must specify two stride dimensions"
            return stride
        else:
            raise TypeError

    def format_pad(self, pad):
        if isinstance(pad, int):
            return [
                    pad - (self.rows_in() - self.kernel_size[0] + 2*pad) % self.stride[0],
                    pad,
                    pad,
                    pad - (self.cols_in() - self.kernel_size[1] + 2*pad) % self.stride[1],
                ]
        elif isinstance(pad, list):
            assert len(pad) == 4, "Must specify four pad dimensions"
            return pad
        else:
            raise TypeError

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
            input_width: int = 16,
            output_width: int = 16,
            weight_width: int = 16,
            acc_width: int = 16,
            biases_width: int = 16,
            has_bias: int = 0 # default to no bias for old configs
        ):

        # initialise parent class
        super().__init__(rows, cols, channels, coarse_in,
                coarse_out, data_width=input_width)

        # save the widths
        self.input_width = input_width
        self.output_width = output_width
        self.weight_width = weight_width
        self.acc_width = acc_width
        self.biases_width = biases_width
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
        self.modules["fork"] = Fork(self.rows_out(), self.cols_out(),
                int(self.channels_in()/self.coarse_in), self.kernel_size, self.coarse_out)
        self.modules["conv"] = Conv(self.rows_out(), self.cols_out(),
                int(self.channels_in()/self.coarse_in), int(self.filters/self.coarse_out),
                self.fine, self.kernel_size, self.groups)
        self.modules["accum"] = Accum(self.rows_out(), self.cols_out(),
                int(self.channels_in()/self.coarse_in),
                int(self.filters/self.coarse_out), self.groups)
        self.modules["glue"] = Glue(self.rows_out(), self.cols_out(), 1,
                int(self.filters/self.coarse_out), self.coarse_in, self.coarse_out)
        self.modules["bias"] = Bias(self.rows_out(), self.cols_out(), 1, self.filters)

        self.update()

    @property
    def kernel_size(self) -> List[int]:
        return self._kernel_size

    @property
    def stride(self) -> List[int]:
        return self._stride

    @property
    def pad(self) -> List[int]:
        return self._pad

    @property
    def pad_top(self) -> int:
        return self._pad[0]

    @property
    def pad_right(self) -> int:
        return self._pad[3]

    @property
    def pad_bottom(self) -> int:
        return self._pad[2]

    @property
    def pad_left(self) -> int:
        return self._pad[1]

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
    def kernel_size(self, val: Union[List[int],int]) -> None:
        self._kernel_size = self.format_kernel_size(val)
        self.update()

    @stride.setter
    def stride(self, val: Union[List[int],int]) -> None:
        self._stride = self.format_stride(val)
        self.update()

    @pad.setter
    def pad(self, val: Union[List[int],int]) -> None:
        self._pad = self.format_pad(val)
        self.pad_top = self._pad[0]
        self.pad_right = self._pad[3]
        self.pad_bottom = self._pad[2]
        self.pad_left = self._pad[1]
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
        return self.modules["sliding_window"].rows_out()

    def cols_out(self) -> int:
        return self.modules["sliding_window"].cols_out()

    def channels_out(self) -> int:
        return self.filters

    def layer_info(self,parameters,batch_size=1):
        Layer.layer_info(self, parameters, batch_size)
        parameters.filters      = self.filters
        parameters.groups       = self.groups
        parameters.coarse_group = self.coarse_group
        parameters.fine         = self.fine
        parameters.kernel_size.extend([self.kernel_size[0], self.kernel_size[1]])
        parameters.stride.extend([self.stride[0], self.stride[1]])
        parameters.pad_top      = self.pad_top
        parameters.pad_right    = self.pad_right
        parameters.pad_bottom   = self.pad_bottom
        parameters.pad_left     = self.pad_left
        parameters.input_width  = self.input_width
        parameters.output_width = self.output_width
        parameters.weight_width = self.weight_width
        parameters.acc_width    = self.acc_width
        parameters.biases_width = self.biases_width
        parameters.has_bias     = self.has_bias

    def update(self):
        # sliding window
        self.modules['sliding_window'].rows     = self.rows
        self.modules['sliding_window'].cols     = self.cols
        self.modules['sliding_window'].channels = self.channels//(self.coarse_in*self.coarse_group)
        self.modules['sliding_window'].data_width   = self.input_width
        # fork
        self.modules['fork'].rows     = self.rows_out()
        self.modules['fork'].cols     = self.cols_out()
        self.modules['fork'].channels = self.channels_in()//(self.coarse_in*self.coarse_group)
        self.modules['fork'].coarse   = self.coarse_out
        self.modules['fork'].data_width     = self.input_width
        # conv
        self.modules['conv'].rows     = self.rows_out()
        self.modules['conv'].cols     = self.cols_out()
        self.modules['conv'].channels = self.channels//(self.coarse_in*self.coarse_group)
        self.modules['conv'].filters  = self.filters//(self.coarse_out*self.coarse_group)
        self.modules['conv'].fine     = self.fine
        self.modules['conv'].groups   = self.groups//self.coarse_group
        self.modules['conv'].data_width     = self.input_width
        self.modules['conv'].weight_width   = self.weight_width
        self.modules['conv'].acc_width      = self.acc_width
        # accum
        self.modules['accum'].rows     = self.rows_out()
        self.modules['accum'].cols     = self.cols_out()
        self.modules['accum'].channels = self.channels//(self.coarse_in*self.coarse_group)
        self.modules['accum'].filters  = self.filters//(self.coarse_out*self.coarse_group)
        self.modules['accum'].groups   = self.groups//self.coarse_group
        self.modules['accum'].data_width    = self.acc_width
        # glue
        self.modules['glue'].rows       = self.rows_out()
        self.modules['glue'].cols       = self.cols_out()
        self.modules['glue'].filters    = self.filters//self.coarse_group
        self.modules['glue'].coarse_in  = self.coarse_in
        self.modules['glue'].coarse_out = self.coarse_out
        self.modules['glue'].data_width = self.output_width
        self.modules['glue'].acc_width  = self.acc_width
        # bias
        self.modules['bias'].rows           = self.rows_out()
        self.modules['bias'].cols           = self.cols_out()
        self.modules['bias'].filters        = self.filters
        self.modules['bias'].data_width     = self.output_width
        self.modules['bias'].biases_width   = self.biases_width

    def get_coarse_group_feasible(self):
        return get_factors(self.groups)

    def get_fine_feasible(self):
        if self.kernel_size[0] != self.kernel_size[1]:
            assert(self.kernel_size[0] == 1 or self.kernel_size[1] == 1)
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

        sw_rsc      = self.modules['sliding_window'].rsc()
        fork_rsc    = self.modules['fork'].rsc()
        conv_rsc    = self.modules['conv'].rsc()
        accum_rsc   = self.modules['accum'].rsc()
        glue_rsc    = self.modules['glue'].rsc()
        bias_rsc    = self.modules['bias'].rsc()

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
                    int(weight_memory_depth),self.weight_width) * \
                self.coarse_in*self.coarse_out*self.coarse_group*self.fine

        # bias usage FIXME depth, FIXME bram usage
        bias_memory_depth = float(self.filters) / float(self.coarse_out)
        biases_bram_usage = bram_memory_resource_model(
                    int(bias_memory_depth),self.biases_width) * self.coarse_out

        # Total
        return {
            "LUT"  :  sw_rsc['LUT']*self.coarse_in*self.coarse_group +
                      fork_rsc['LUT']*self.coarse_in*self.coarse_group +
                      conv_rsc['LUT']*self.coarse_in*self.coarse_out*self.coarse_group +
                      accum_rsc['LUT']*self.coarse_in*self.coarse_out*self.coarse_group +
                      glue_rsc['LUT']*self.coarse_group +
                      bias_rsc['LUT']*self.coarse_out,
            "FF"   :  sw_rsc['FF']*self.coarse_in*self.coarse_group +
                      fork_rsc['FF']*self.coarse_in*self.coarse_group +
                      conv_rsc['FF']*self.coarse_in*self.coarse_out*self.coarse_group +
                      accum_rsc['FF']*self.coarse_in*self.coarse_out*self.coarse_group +
                      glue_rsc['FF']*self.coarse_group +
                      bias_rsc['FF']*self.coarse_out,
            "BRAM" :  sw_rsc['BRAM']*self.coarse_in*self.coarse_group +
                      fork_rsc['BRAM']*self.coarse_in*self.coarse_group +
                      conv_rsc['BRAM']*self.coarse_in*self.coarse_out*self.coarse_group +
                      accum_rsc['BRAM']*self.coarse_out*self.coarse_group +
                      glue_rsc['BRAM']*self.coarse_group +
                      bias_rsc['BRAM']*self.coarse_out +
                      weights_bram_usage +
                      biases_bram_usage,
            "DSP" :   sw_rsc['DSP']*self.coarse_in*self.coarse_group +
                      fork_rsc['DSP']*self.coarse_in*self.coarse_group +
                      conv_rsc['DSP']*self.coarse_in*self.coarse_out*self.coarse_group +
                      accum_rsc['DSP']*self.coarse_in*self.coarse_out*self.coarse_group +
                      glue_rsc['DSP']*self.coarse_group +
                      bias_rsc['DSP']*self.coarse_out
        }

    def visualise(self,name): #TODO for biases
        cluster = pydot.Cluster(name,label=name)
        nodes_in = []
        nodes_out = []

        for g in range(self.coarse_group):
            for i in range(self.coarse_in):
                cluster.add_node(pydot.Node(
                    "_".join([name,"sw",str(g*self.coarse_in+i)]), label="sw" ))

            for i in range(self.coarse_in):
                cluster.add_node(pydot.Node(
                    "_".join([name,"fork",str(g*self.coarse_in+i)]), label="fork" ))
                cluster.add_edge(pydot.Edge(
                    "_".join([name,"sw",str(g*self.coarse_in+i)]),
                    "_".join([name,"fork",str(i)]) ))

            for i in range(self.coarse_in):
                for j in range(self.coarse_out):
                    cluster.add_node(pydot.Node(
                        "_".join([name,"conv",str(g*self.coarse_in+i),
                            str(g*self.coarse_out+j)]), label="conv" ))
                    cluster.add_edge(pydot.Edge(
                        "_".join([name,"fork",str(g*self.coarse_in+i)]),
                        "_".join([name,"conv",str(g*self.coarse_in+i),str(g*self.coarse_out+j)]) ))

            for i in range(self.coarse_in):
                for j in range(self.coarse_out):
                    cluster.add_node(pydot.Node(
                        "_".join([name,"glue",str(g*self.coarse_out+j)]), label="+" ))
                    cluster.add_node(pydot.Node(
                        "_".join([name,"accum",str(g*self.coarse_in+i),
                            str(g*self.coarse_out+j)]), label="accum" ))
                    cluster.add_edge(pydot.Edge(
                        "_".join([name,"conv" ,str(g*self.coarse_in+i),str(g*self.coarse_out+j)]),
                        "_".join([name,"accum",str(g*self.coarse_in+i),str(g*self.coarse_out+j)])
                        ))
                    cluster.add_edge(pydot.Edge(
                        "_".join([name,"accum",str(g*self.coarse_in+i),str(g*self.coarse_out+j)]),
                        "_".join([name,"glue",str(g*self.coarse_out+j)]) ))

            # get nodes in and out
            for i in range(self.coarse_in):
                nodes_in.append("_".join([name,"sw",str(g*self.coarse_in+i)]))

            for j in range(self.coarse_out):
                nodes_out.append("_".join([name,"glue",str(g*self.coarse_out+j)]))

        return cluster, nodes_in, nodes_out

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
                stride=self.stride, padding=self.pad[0], groups=self.groups)

        # update weights
        convolution_layer.weight = torch.nn.Parameter(torch.from_numpy(weights))

        # update bias
        convolution_layer.bias = torch.nn.Parameter(torch.from_numpy(bias))

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return convolution_layer(torch.from_numpy(data)).detach().numpy()

