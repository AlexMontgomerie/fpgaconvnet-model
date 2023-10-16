"""
The split/fork/branch layer.
Takes one stream input and outputs several streams using the fork module.
"""

from dataclasses import dataclass, field
from typing import Any

import pydot

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.layers import MultiPortLayer
from fpgaconvnet.models.modules import Fork


@dataclass(kw_only=True)
class SplitLayer(MultiPortLayer):
    coarse: int = 1
    ports_out: int = 1
    backend: str = "chisel"
    regression_model: str = "linear_regression"

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        # backend flag
        assert (self.backend in ["hls", "chisel"], f"{self.backend} is an invalid backend")

        # regression model
        assert(self.regression_model in ["linear_regression", "xgboost"],
                f"{self.regression_model} is an invalid regression model")

        self.mem_bw_out = [100.0/self.ports_out] * self.ports_out

        # init modules
        #One fork module, fork coarse_out corresponds to number of layer output ports
        self.modules["fork"] = Fork( self.rows_in(), self.cols_in(),
                self.channels_in(), 1, self.ports_out, backend=self.backend, regression_model=self.regression_model)

        # update the modules
        self.update()

    def __setattr__(self, name: str, value: Any) -> None:

        if not hasattr(self, "is_init"):
            super().__setattr__(name, value)
            return

        match name:
            case "coarse":
                assert(value in self.get_coarse_in_feasible())
                super().__setattr__("coarse", value)
                self.update()
            case "coarse_in":
                assert(value in self.get_coarse_in_feasible())
                super().__setattr__("coarse_in", [value])
                self.update()
            case "coarse_out":
                assert(value in self.get_coarse_out_feasible())
                super().__setattr__("coarse_out", [value]*self.ports_out)
                self.update()
            case _:
                super().__setattr__(name, value)

    def streams_in(self, port_index=0):
        """
        Returns
        -------
        int
            number of parallel streams into the layer.
        """
        assert(port_index < self.ports_in)
        return self._coarse

    def rows_out(self, port_index=0) -> int:
        return self.rows[0]

    def cols_out(self, port_index=0) -> int:
        return self.cols[0]

    def channels_out(self, port_index=0) -> int:
        return self.channels[0]

    def layer_info(self,parameters,batch_size=1):
        MultiPortLayer.layer_info(self, parameters, batch_size)
        parameters.coarse = self.coarse
        parameters.rows_in      = self.rows_in()
        parameters.cols_in      = self.cols_in()
        parameters.channels_in  = self.channels_in()
        parameters.rows_out     = self.rows_out()
        parameters.cols_out     = self.cols_out()
        parameters.channels_out = self.channels_out()
        # remove the repeated rows, cols and channels
        del parameters.rows_in_array[:]
        del parameters.cols_in_array[:]
        del parameters.channels_in_array[:]
        del parameters.rows_out_array[:]
        del parameters.cols_out_array[:]
        del parameters.channels_out_array[:]

    def update(self):
        # fork
        self.modules['fork'].rows     = self.rows_in()
        self.modules['fork'].cols     = self.cols_in()
        self.modules['fork'].channels = self.channels_in()//self.coarse
        self.modules['fork'].coarse   = self.ports_out

    def resource(self):

        # get module resources
        fork_rsc = self.modules['fork'].rsc()

        #Total
        return {
            "LUT"   :   fork_rsc['LUT']*self.coarse,
            "FF"    :   fork_rsc['FF']*self.coarse,
            "BRAM"  :   fork_rsc['BRAM']*self.coarse,
            "DSP"   :   fork_rsc['DSP']*self.coarse
        }

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in self.coarse_in:
            cluster.add_node(pydot.Node( "_".join([name,"split",str(i)]), label="split" ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"split",str(i)]) for i in range(self.coarse) ]
        nodes_out = [ "_".join([name,"split",str(i)]) for i in range(self.ports_out) ]

        return cluster, nodes_in, nodes_out

    def functional_model(self, data, batch_size=1):

        assert data.shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels_in(), "ERROR (data): invalid channel dimension"

        return [data for _ in range(self.ports_out)]
