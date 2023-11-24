import math
import tempfile
from dataclasses import dataclass, field

from typing import Any

import numpy as np
import pydot

from fpgaconvnet.data_types import FixedPoint

from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model

# from fpgaconvnet.models.modules import ShiftScale
from fpgaconvnet.models.layers import Layer

@dataclass(kw_only=True)
class BatchNormLayer(Layer):
    coarse: int = 1

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        self.input_t = FixedPoint(32, 0)
        self.scale_t = FixedPoint(32, 0)
        self.shift_t = FixedPoint(8, 0)
        self.output_t = FixedPoint(9, 0)

        # init variables
        self.scale_layer = None

        # modules
        # self.modules["shift_scale"] = ShiftScale(self.rows, self.cols, self.channels)

        # update modules
        self.update()

    def __setattr__(self, name: str, value: Any) -> None:

        if not hasattr(self, "is_init"):
            super().__setattr__(name, value)
            return

        match name:
            case "coarse" | "coarse_in" | "coarse_out":
                assert(value in self.get_coarse_in_feasible())
                assert(value in self.get_coarse_out_feasible())
                super().__setattr__("coarse_in", value)
                super().__setattr__("coarse_out", value)
                super().__setattr__("coarse", value)
                self.update()

            case _:
                super().__setattr__(name, value)

    def layer_info(self,parameters, batch_size=1):
        Layer.layer_info(self, parameters, batch_size)
        parameters.coarse = self.coarse
        parameters.scale.extend(self.scale)
        parameters.shift.extend(self.shift)
        self.input_t.to_protobuf(parameters.input_t)
        self.output_t.to_protobuf(parameters.output_t)
        self.scale_t.to_protobuf(parameters.scale_t)
        self.shift_t.to_protobuf(parameters.shift_t)

    def update(self):
        # batch norm
        pass
        # self.modules['shift_scale'].rows     = self.rows_in()
        # self.modules['shift_scale'].cols     = self.cols_in()
        # self.modules['shift_scale'].channels = self.channels_in()//self.coarse

    def resource(self):

        # get batch norm layer usage
        # shift_scale_rsc      = self.modules['shift_scale'].rsc()

        # get bram usage of scale, bias parameter
        # weights_bram_usage = bram_array_resource_model(
        #         self.channels//self.coarse, self.data_width, 'memory')*self.coarse*2
        # Total
        return {
            "LUT"  :  0,
            "FF"   :  0,
            "BRAM" :  0,
            "DSP" :   2*self.coarse
        }

    def functional_model(self,data,gamma,beta,batch_size=1):

        import torch

        assert data.shape[0] == self.rows    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR (data): invalid channel dimension"

        assert gamma.shape[0] == self.channels , "ERROR (weights): invalid filter dimension"
        assert beta.shape[0]  == self.channels , "ERROR (weights): invalid filter dimension"

        # instantiate batch norm layer
        batch_norm_layer = torch.nn.BatchNorm2d(self.channels, track_running_stats=False)

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return batch_norm_layer(torch.from_numpy(data))

