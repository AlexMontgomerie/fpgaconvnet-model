import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.layers.ConvolutionPointwiseSparseLayer import ConvolutionPointwiseSparseLayer

class InnerProductSparseLayer(ConvolutionPointwiseSparseLayer):
    def __init__(
            self,
            filters: int,
            rows: int,
            cols: int,
            channels: int,
            coarse_in: int = 1,
            coarse_out: int = 1,
            fine: int  = 1,
            input_t: FixedPoint = FixedPoint(16,8),
            output_t: FixedPoint = FixedPoint(16,8),
            weight_t: FixedPoint = FixedPoint(16,8),
            acc_t: FixedPoint = FixedPoint(32,16),
            has_bias: int = 0, # default to no bias for old configs
            sparsity: list = [],
            clusters: int = 1,
            backend: str = "chisel", # default to no bias for old configs
            regression_model: str = "linear_regression",  
            stream_weights: int = 0,
            use_uram: bool = False,
            input_compression_ratio: list = [1.0],
            output_compression_ratio: list = [1.0],
            weight_compression_ratio: list = [1.0]
        ):

        # initialise parent class
        super().__init__(
            filters,
            rows,
            cols,
            channels,
            coarse_in=coarse_in,
            coarse_out=coarse_out,
            fine=fine,
            input_t=input_t,
            output_t=output_t,
            weight_t=weight_t,
            acc_t=acc_t,
            has_bias=has_bias, # default to no bias for old configs
            sparsity=sparsity,
            clusters=clusters,
            backend=backend,
            regression_model=regression_model,
            stream_weights=stream_weights,
            use_uram=use_uram,
            input_compression_ratio=input_compression_ratio,
            output_compression_ratio=output_compression_ratio,
            weight_compression_ratio=weight_compression_ratio
        )

    def functional_model(self,data,weights,bias,batch_size=1):
        import torch

        assert data.shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels_in(), "ERROR (data): invalid channel dimension"

        assert weights.shape[0] == self.filters ,   "ERROR (weights): invalid filter dimension"
        assert weights.shape[1] == self.rows_in()*self.cols_in()*self.channels_in(),\
                                                    "ERROR (weights): invalid channel dimension"


        # instantiate inner product layer
        inner_product_layer = torch.nn.Linear(
                self.channels_in()*self.rows_in()*self.cols_in(), self.filters)#, bias=False)

        # update weights
        inner_product_layer.weight = torch.nn.Parameter(torch.from_numpy(weights))

        # update bias
        inner_product_layer.bias = torch.nn.Parameter(torch.from_numpy(bias))

        # return output featuremap
        data = np.moveaxis(data, -1, 0).flatten()
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return inner_product_layer(torch.from_numpy(data)).detach().numpy()

