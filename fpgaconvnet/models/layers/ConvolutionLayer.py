import numpy as np
import math
import pydot
import torch
from typing import Union, List

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2

from fpgaconvnet.models.layers.utils import get_factors

from fpgaconvnet.models.layers import Layer

class ConvolutionLayerBase(Layer):

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
        parameters.has_bias     = self.has_bias
        self.input_t.to_protobuf(parameters.input_t)
        self.output_t.to_protobuf(parameters.output_t)
        self.weight_t.to_protobuf(parameters.weight_t)
        self.acc_t.to_protobuf(parameters.acc_t)
        parameters.data_t.Clear()

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

