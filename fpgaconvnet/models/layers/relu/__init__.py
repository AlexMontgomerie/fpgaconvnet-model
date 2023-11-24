from typing import ClassVar

from fpgaconvnet.models.layers import LayerBase
from fpgaconvnet.models.layers.traits import Layer2D, Layer3D
from .base import ReLULayerBase, ReLULayerChiselMixin, ReLULayerHLSMixin

class ReLULayerChisel2D(ReLULayerChiselMixin, Layer2D, ReLULayerBase):
    register: ClassVar[bool] = True

class ReLULayerChisel3D(ReLULayerChiselMixin, Layer3D, ReLULayerBase):
    register: ClassVar[bool] = True

class ReLULayerHLS2D(ReLULayerHLSMixin, Layer2D, ReLULayerBase):
    register: ClassVar[bool] = True

class ReLULayerHLS3D(ReLULayerHLSMixin, Layer3D, ReLULayerBase):
    register: ClassVar[bool] = True
