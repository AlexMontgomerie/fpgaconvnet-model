from dataclasses import dataclass, field
import numpy as np

@dataclass(kw_only=True)
class HardswishLayerTrait2D:

    def get_hardswish_parameters(self):
        return {
            "rows"      : self.rows_in(),
            "cols"      : self.cols_in(),
            "channels"  : self.channels_in()//self.streams_in(),
            "data_width": self.input_t.width,
        }

    def functional_model(self,data,batch_size=1):
        import torch

        assert data.shape[0] == self.rows_in()    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels_in(), "ERROR: invalid channel dimension"

        # instantiate hardswish layer
        hardswish_layer = torch.nn.Hardswish()

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return hardswish_layer(torch.from_numpy(data)).detach().numpy()

@dataclass(kw_only=True)
class HardswishLayerTrait3D(HardswishLayerTrait2D):

    def get_hardswish_parameters(self):
        param = super().get_hardswish_parameters()
        param["depth"] = self.depth_in()
        return param

    def functional_model(self,data,batch_size=1):
        import torch

        assert data.shape[0] == self.rows_in()    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth_in()   , "ERROR: invalid depth dimension"
        assert data.shape[3] == self.channels_in(), "ERROR: invalid channel dimension"

        # instantiate hardswish layer
        hardswish_layer = torch.nn.Hardswish()

        # return output featuremap
        data = np.moveaxis(data, [-1, -2], [0, 1])
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return hardswish_layer(torch.from_numpy(data)).detach().numpy()


