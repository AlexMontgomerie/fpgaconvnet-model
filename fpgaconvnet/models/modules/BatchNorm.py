"""
.. figure:: ../../../figures/batch_norm_diagram.png
"""
import numpy as np
import math
import os
from dataclasses import dataclass, field

from fpgaconvnet.models.modules import Module, MODULE_FONTSIZE

@dataclass
class BatchNorm(Module):

    def __post_init__(self):
        pass
        # # load the resource model coefficients
        # self.rsc_coef["LUT"] = np.load(
        #         os.path.join(os.path.dirname(__file__),
        #         "../../coefficients/linear_regression/chisel/relu_lut.npy"))
        # self.rsc_coef["FF"] = np.load(
        #         os.path.join(os.path.dirname(__file__),
        #         "../../coefficients/linear_regression/chisel/relu_ff.npy"))
        # self.rsc_coef["BRAM"] = np.load(
        #         os.path.join(os.path.dirname(__file__),
        #         "../../coefficients/linear_regression/chisel/relu_bram.npy"))
        # self.rsc_coef["DSP"] = np.load(
        #         os.path.join(os.path.dirname(__file__),
        #         "../../coefficients/linear_regression/chisel/relu_dsp.npy"))

    def rsc(self):
        return {
          "LUT"  : 0, #int(np.dot(self.utilisation_model(), self.rsc_coef[0])),
          "BRAM" : 0,
          "DSP"  : 1,
          "FF"   : 0 #int(np.dot(self.utilisation_model(), self.rsc_coef[3])),
        }

    def functional_model(self, data, scale, shift):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"

        out = np.ndarray((
            self.rows,
            self.cols,
            self.channels),dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = scale[index[2]] * ( data[index] + shift[index[2]] )

        return out


