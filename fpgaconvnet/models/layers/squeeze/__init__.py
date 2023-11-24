from typing import ClassVar
from dataclasses import dataclass

from fpgaconvnet.models.layers import LayerBase
from fpgaconvnet.models.layers.traits import Layer2D, Layer3D
from .base import SqueezeLayerBase, SqueezeLayerChiselMixin, SqueezeLayerHLSMixin

@dataclass(kw_only=True)
class SqueezeLayerChisel2D(SqueezeLayerChiselMixin, Layer2D, SqueezeLayerBase):
    register: ClassVar[bool] = True

@dataclass(kw_only=True)
class SqueezeLayerChisel3D(SqueezeLayerChiselMixin, Layer3D, SqueezeLayerBase):
    register: ClassVar[bool] = True

@dataclass(kw_only=True)
class SqueezeLayerHLS2D(SqueezeLayerHLSMixin, Layer2D, SqueezeLayerBase):
    register: ClassVar[bool] = True

@dataclass(kw_only=True)
class SqueezeLayerHLS3D(SqueezeLayerHLSMixin, Layer3D, SqueezeLayerBase):
    register: ClassVar[bool] = True
