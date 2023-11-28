from typing import ClassVar
from dataclasses import dataclass

from fpgaconvnet.models.layers import LayerBase
from fpgaconvnet.models.layers.traits import Layer2D, Layer3D
from .base import PoolingLayerBase, PoolingLayerChiselMixin, PoolingLayerHLSMixin, PoolingLayer2DMixin, PoolingLayer3DMixin

@dataclass(kw_only=True)
class PoolingLayerChisel2D(PoolingLayerChiselMixin, PoolingLayer2DMixin):
    register: ClassVar[bool] = True

@dataclass(kw_only=True)
class PoolingLayerChisel3D(PoolingLayerChiselMixin, PoolingLayer3DMixin, PoolingLayerBase):
    register: ClassVar[bool] = True

@dataclass(kw_only=True)
class PoolingLayerHLS2D(PoolingLayerHLSMixin, PoolingLayer2DMixin, PoolingLayerBase):
    register: ClassVar[bool] = True

@dataclass(kw_only=True)
class PoolingLayerHLS3D(PoolingLayerHLSMixin, PoolingLayer3DMixin, PoolingLayerBase):
    register: ClassVar[bool] = True
