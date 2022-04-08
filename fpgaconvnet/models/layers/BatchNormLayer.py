import numpy as np
import math
import tempfile
import pydot

from fpgaconvnet.tools.resource_model import bram_memory_resource_model

from fpgaconvnet.models.modules import BatchNorm
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
        self.modules["batch_norm"] = BatchNorm(self.rows, self.cols, self.channels)

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
        self.modules['batch_norm'].rows     = self.rows_in()
        self.modules['batch_norm'].cols     = self.cols_in()
        self.modules['batch_norm'].channels = self.channels_in()//self.coarse

    def resource(self):

        # get batch norm layer usage
        bn_rsc      = self.modules['batch_norm'].rsc()

        # get bram usage of scale parameter
        weights_bram_usage = bram_memory_resource_models(self.channels//self.coarse,self.data_width)*self.coarse

        # Total
        return {
            "LUT"  :  bn_rsc['LUT']*self.coarse,
            "FF"   :  bn_rsc['FF']*self.coarse,
            "BRAM" :  bn_rsc['BRAM']*self.coarse + weights_bram_usage,
            "DSP" :   bn_rsc['DSP']*self.coarse
        }

    def functional_model(self,data,gamma,beta,batch_size=1):


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

