from typing import ClassVar
from dataclasses import dataclass

from fpgaconvnet.models.layers import LayerBase
from fpgaconvnet.models.layers.traits import Layer2D, Layer3D
from .base import ResizeLayerBase, ResizeLayerChiselMixin, ResizeLayer2DMixin

@dataclass(kw_only=True)
class ResizeLayerChisel2D(ResizeLayerChiselMixin, ResizeLayer2DMixin, ResizeLayerBase):
    register: ClassVar[bool] = True

