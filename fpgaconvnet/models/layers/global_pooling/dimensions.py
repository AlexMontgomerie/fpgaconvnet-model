from dataclasses import dataclass, field
import numpy as np

@dataclass(kw_only=True)
class GlobalPoolingLayerTrait2D:

    def get_global_pooling_parameters(self):
        return {
            "rows"      : self.rows_in(),
            "cols"      : self.cols_in(),
            "channels"  : self.channels_in()//self.coarse,
            "data_width": self.input_t.width,
            "acc_width": self.acc_t.width,
        }

    def functional_model(self, data, batch_size=1):

        assert data.shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels_in(), "ERROR (data): invalid channel dimension"

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return np.average(data, axis=(2,3))

@dataclass(kw_only=True)
class GlobalPoolingLayerTrait3D(GlobalPoolingLayerTrait2D):

    def depth_out(self) -> int:
        return 1

    def get_global_pooling_parameters(self):
        param = super().get_global_pooling_parameters()
        param["depth"] = self.depth_in()
        return param

    def functional_model(self, data, batch_size=1):

        assert data.shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.depth_in()   , "ERROR (data): invalid depth dimension"
        assert data.shape[3] == self.channels_in(), "ERROR (data): invalid channel dimension"

        # return output featuremap
        data = np.moveaxis(data, [-1, -2], [0, 1])
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return np.average(data, axis=(2,3,4))

