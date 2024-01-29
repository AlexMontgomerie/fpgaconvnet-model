from typing import ClassVar
from dataclasses import dataclass

from .base import InnerProductLayerBase, InnerProductLayer2DMixin, InnerProductLayer3DMixin
from .backend import InnerProductLayerChiselMixin, InnerProductLayerHLSMixin

@dataclass(kw_only=True)
class InnerProductLayerChisel2D(InnerProductLayerChiselMixin, InnerProductLayer2DMixin):
    register: ClassVar[bool] = True

@dataclass(kw_only=True)
class InnerProductLayerChisel3D(InnerProductLayerChiselMixin, InnerProductLayer3DMixin):
    register: ClassVar[bool] = True

@dataclass(kw_only=True)
class InnerProductLayerHLS2D(InnerProductLayerHLSMixin, InnerProductLayer2DMixin):
    register: ClassVar[bool] = True

@dataclass(kw_only=True)
class InnerProductLayerHLS3D(InnerProductLayerHLSMixin, InnerProductLayer3DMixin):
    register: ClassVar[bool] = True

