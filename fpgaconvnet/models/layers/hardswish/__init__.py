from typing import ClassVar
from dataclasses import dataclass

from fpgaconvnet.models.layers import LayerBase
from fpgaconvnet.models.layers.traits import Layer2D, Layer3D
from .base import HardswishLayerBase, HardswishLayerChiselMixin, HardswishLayerHLSMixin

@dataclass(kw_only=True)
class HardswishLayerChisel2D(HardswishLayerChiselMixin, Layer2D, HardswishLayerBase):
    register: ClassVar[bool] = True

@dataclass(kw_only=True)
class HardswishLayerChisel3D(HardswishLayerChiselMixin, Layer3D, HardswishLayerBase):
    register: ClassVar[bool] = True

@dataclass(kw_only=True)
class HardswishLayerHLS2D(HardswishLayerHLSMixin, Layer2D, HardswishLayerBase):
    register: ClassVar[bool] = True

@dataclass(kw_only=True)
class HardswishLayerHLS3D(HardswishLayerHLSMixin, Layer3D, HardswishLayerBase):
    register: ClassVar[bool] = True
