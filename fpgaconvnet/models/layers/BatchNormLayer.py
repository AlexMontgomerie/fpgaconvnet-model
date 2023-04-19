import numpy as np
import math
import tempfile
import pydot

from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model

from fpgaconvnet.models.modules import ShiftScale
from fpgaconvnet.models.layers import Layer

class BatchNormLayer(Layer):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            coarse: int = 1,
            data_width: int = 16
        ):

        super().__init__(self,rows, cols, channels, coarse,
                coarse, data_width=data_width)

        # save parameters
        self._coarse = coarse

        # init variables
        self.scale_layer = None

        # modules
        self.modules["shift_scale"] = ShiftScale(self.rows, self.cols, self.channels)

        # update modules
        self.update()

    @property
    def coarse(self) -> int:
        return self._coarse

    @property
    def coarse_in(self) -> int:
        return self._coarse

    @property
    def coarse_out(self) -> int:
        return self._coarse

    @coarse.setter
    def coarse(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = val
        self.coarse_out = val
        self.update()

    @coarse_in.setter
    def coarse_in(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = val
        self._coarse_out = val
        self.update()

    @coarse_out.setter
    def coarse_out(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = val
        self._coarse_out = val
        self.update()

    def layer_info(self,parameters,batch_size=1):
        Layer.layer_info(self, parameters, batch_size)
        parameters.coarse = self.coarse

    def update(self):
        # batch norm
        self.modules['shift_scale'].rows     = self.rows_in()
        self.modules['shift_scale'].cols     = self.cols_in()
        self.modules['shift_scale'].channels = self.channels_in()//self.coarse

    def resource(self):

        # get batch norm layer usage
        shift_scale_rsc      = self.modules['shift_scale'].rsc()

        # get bram usage of scale, bias parameter
        weights_bram_usage = bram_array_resource_model(self.channels//self.coarse, self.data_width, 'memory')*self.coarse*2
        # Total
        return {
            "LUT"  :  shift_scale_rsc['LUT']*self.coarse,
            "FF"   :  shift_scale_rsc['FF']*self.coarse,
            "BRAM" :  shift_scale_rsc['BRAM']*self.coarse + weights_bram_usage,
            "DSP" :   shift_scale_rsc['DSP']*self.coarse
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

