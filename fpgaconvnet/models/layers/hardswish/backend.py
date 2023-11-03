
import importlib
import math
from typing import Union, List
from dataclasses import dataclass, field

import pydot
import numpy as np
from dacite import from_dict

from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.models.modules import Hardswish

@dataclass(kw_only=True)
class HardswishLayerTraitHLS:

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        # create all the modules
        self.modules["hardswish"] = Hardswish(self.rows_in(), self.cols_in(),
                self.channels_in()//self.coarse, backend="hls", regression_model=self.regression_model)

        # update modules
        self.update_modules()

    def update_modules(self):

        # update the hardswish module
        param = self.get_hardswish_parameters()
        for p, v in param.items():
            setattr(self.modules["hardswish"], p, v)


    def resource(self):

        # get hardswish3d resources
        hardswish_rsc = self.modules['hardswish'].rsc()

        # Total
        return {
            "LUT"  :  hardswish_rsc['LUT']*self.coarse,
            "FF"   :  hardswish_rsc['FF']*self.coarse,
            "BRAM" :  hardswish_rsc['BRAM']*self.coarse,
            "DSP" :   hardswish_rsc['DSP']*self.coarse,
        }

@dataclass(kw_only=True)
class HardswishLayerTraitChisel:

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        # create all the modules
        self.modules["hardswish"] = Hardswish(self.rows_in(), self.cols_in(),
                self.channels_in()//self.coarse, backend="chisel", regression_model=self.regression_model)

        # update modules
        self.update_modules()

    def update_modules(self):

        # update the hardswish module
        param = self.get_hardswish_parameters()
        for p, v in param.items():
            setattr(self.modules["hardswish"], p, v)


    def resource(self):

        # get hardswish3d resources
        hardswish_rsc = self.modules['hardswish'].rsc()

        # Total
        return {
            "LUT"  :  hardswish_rsc['LUT']*self.coarse,
            "FF"   :  hardswish_rsc['FF']*self.coarse,
            "BRAM" :  hardswish_rsc['BRAM']*self.coarse,
            "DSP" :   hardswish_rsc['DSP']*self.coarse,
        }

