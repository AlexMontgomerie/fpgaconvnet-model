from typing import ClassVar
from dataclasses import dataclass

from fpgaconvnet.models.layers import LayerBase
from fpgaconvnet.models.layers.traits import Layer2D, Layer3D
from .base import ReLULayerBase, ReLULayerChiselMixin, ReLULayerHLSMixin

@dataclass(kw_only=True)
class ReLULayerChisel2D(ReLULayerChiselMixin, Layer2D, ReLULayerBase):
    register: ClassVar[bool] = True

@dataclass(kw_only=True)
class ReLULayerChisel3D(ReLULayerChiselMixin, Layer3D, ReLULayerBase):
    register: ClassVar[bool] = True

@dataclass(kw_only=True)
class ReLULayerHLS2D(ReLULayerHLSMixin, Layer2D, ReLULayerBase):
    register: ClassVar[bool] = True

@dataclass(kw_only=True)
class ReLULayerHLS3D(ReLULayerHLSMixin, Layer3D, ReLULayerBase):
    register: ClassVar[bool] = True
