from typing import ClassVar
from dataclasses import dataclass

from fpgaconvnet.models.layers import LayerBase
from fpgaconvnet.models.layers.traits import Layer2D, Layer3D
from .base import GlobalPoolingLayerBase, GlobalPoolingLayerChiselMixin, GlobalPoolingLayer2DMixin, GlobalPoolingLayer3DMixin

@dataclass(kw_only=True)
class GlobalPoolingLayerChisel2D(GlobalPoolingLayerChiselMixin, GlobalPoolingLayer2DMixin, GlobalPoolingLayerBase):
    register: ClassVar[bool] = True

@dataclass(kw_only=True)
class GlobalPoolingLayerChisel3D(GlobalPoolingLayerChiselMixin, GlobalPoolingLayer3DMixin, GlobalPoolingLayerBase):
    register: ClassVar[bool] = True

