from typing import ClassVar
from dataclasses import dataclass

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port

@dataclass(kw_only=True)
class ShiftScaleChisel(ModuleChiselBase):

    # # hardware parameters
    # data_t: FixedPoint = FixedPoint(16, 8)
    # input_buffer_depth: int = 0
    # output_buffer_depth: int = 0

    # class variables
    name: ClassVar[str] = "shift_scale"
    register: ClassVar[bool] = True

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


