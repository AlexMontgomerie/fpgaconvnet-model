from dataclasses import dataclass

import pydot
import numpy as np

# from fpgaconvnet.models.layers import Layer3D
# from fpgaconvnet.models.modules import Squeeze3D

@dataclass(kw_only=True)
# class SqueezeLayer3D(Layer3D):
class SqueezeLayer3D:
    coarse_in: int
    coarse_out: int
    backend: str = "chisel"
    regression_model: str = "linear_regression"

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        # backend flag
        assert(self.backend in ["chisel"], f"{self.backend} is an invalid backend")

        # regression model
        assert(self.regression_model in ["linear_regression", "xgboost"],
               f"{self.regression_model} is an invalid regression model")

        # initialise modules
        self.modules["squeeze3d"] = Squeeze3D(self.rows, self.cols, self.depth, self.channels, self.coarse_in, self.coarse_out, backend=self.backend, regression_model=self.regression_model)

        self.update()

    def layer_info(self,parameters,batch_size=1):
        Layer3D.layer_info(self, parameters, batch_size)

    def update(self):
        self.modules["squeeze3d"].rows = self.rows
        self.modules["squeeze3d"].cols = self.cols
        self.modules["squeeze3d"].depth = self.depth
        self.modules["squeeze3d"].channels = self.channels
        self.modules["squeeze3d"].coarse_in = self.coarse_in
        self.modules["squeeze3d"].coarse_out = self.coarse_out
        self.modules["squeeze3d"].data_width = self.data_t.width

    def resource(self):

        # get squeeze resources
        squeeze_rsc = self.modules['squeeze3d'].rsc()

        # Total
        return {
            "LUT"  :  squeeze_rsc['LUT'],
            "FF"   :  squeeze_rsc['FF'],
            "BRAM" :  squeeze_rsc['BRAM'],
            "DSP" :   squeeze_rsc['DSP'],
        }

    def visualise(self,name):

        # create layer cluster
        cluster = pydot.Cluster(name, label=name,
                style="dashed", bgcolor="moccasin")

        # add squeeze module
        squeeze_name = "_".join([name,"squeeze3d"])
        cluster.add_node(self.modules["squeeze3d"].visualise(squeeze_name))

        # get nodes in and out
        nodes_in  = [ squeeze_name for i in range(self.streams_in()) ]
        nodes_out = [ squeeze_name for i in range(self.streams_out()) ]

        # return module
        return cluster, nodes_in, nodes_out

    def functional_model(self,data,batch_size=1):

        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth   , "ERROR: invalid depth dimension"
        assert data.shape[3] == self.channels, "ERROR: invalid channel dimension"

        # return output featuremap
        data = np.moveaxis(data, [-1, -2], [0, 1])
        return np.repeat(data[np.newaxis,...], batch_size, axis=0)

