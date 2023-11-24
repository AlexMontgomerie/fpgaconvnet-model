from dataclasses import dataclass
from typing import Any, List

from fpgaconvnet.models.layers import MultiPortLayer
from fpgaconvnet.models.layers.utils import get_factors
# from fpgaconvnet.models.modules import Concat


@dataclass(kw_only=True)
class ConcatLayer(MultiPortLayer):
    channels: List[int]
    ports_in: int = 1
    coarse: int = 1
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

        self.mem_bw_in = [100.0] * self.ports_in

        # init modules
        self.modules = {
            # "concat" : Concat(self.rows_in(), self.cols_in(), self.channels, self.ports_in,
            #     backend=self.backend, regression_model=self.regression_model),
            "concat" : Concat(self.rows_in(), self.cols_in(), self.channels, self.ports_in,
                backend=self.backend, regression_model=self.regression_model),
        }

        # update the layer
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
                super().__setattr__("coarse_in", [value]*self.ports_in)
                self.update()
            case "coarse_out":
                assert(value in self.get_coarse_out_feasible())
                super().__setattr__("coarse_out", [value])
                self.update()
            case _:
                super().__setattr__(name, value)

    def channels_out(self, port_index=0):
        assert port_index == 0, "ConcatLayer only has a single output port"
        return sum(self.channels)

    def get_operations(self):
        raise NotImplementedError

    def rates_in(self, port_index=0):
        assert port_index < self.ports_in
        return self.modules["concat"].rate_in(port_index)

    def rates_out(self, port_index=0):
        assert port_index == 0, "ConcatLayer only has a single output port"
        return self.modules["concat"].rate_out()

    def get_coarse_in_feasible(self, port_index=0):
        assert(port_index < self.ports_in)
        factors = set( get_factors(self.channels_in(0)) )
        for i in range(self.ports_in):
            factors &= set( get_factors(self.channels_in(i)) )
        return list(factors)

    def get_coarse_out_feasible(self, port_index=0):
        assert(port_index < self.ports_out)
        return self.get_coarse_in_feasible()

    def update(self):
        # concat
        self.modules["concat"].rows     = self.rows_in()
        self.modules["concat"].cols     = self.cols_in()
        self.modules["concat"].channels = self.channels
        self.modules["concat"].ports_in = self.ports_in

    def layer_info(self,parameters,batch_size=1):
        MultiPortLayer.layer_info(self, parameters, batch_size)
        parameters.rows_in      = self.rows_in()
        parameters.cols_in      = self.cols_in()
        parameters.rows_out     = self.rows_out()
        parameters.cols_out     = self.cols_out()
        parameters.channels_out = self.channels_out()
        parameters.coarse       = self._coarse
       # remove the repeated rows, cols and channels
        del parameters.rows_in_array[:]
        del parameters.cols_in_array[:]
        del parameters.rows_out_array[:]
        del parameters.cols_out_array[:]
        del parameters.channels_out_array[:]

