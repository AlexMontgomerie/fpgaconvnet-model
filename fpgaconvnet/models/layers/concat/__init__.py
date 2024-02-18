from typing import ClassVar
from dataclasses import dataclass

from fpgaconvnet.models.layers import LayerBase
from .base import ConcatLayerBase, ConcatLayerChiselMixin, ConcatLayer2DMixin

@dataclass(kw_only=True)
class ConcatLayerChisel2D(ConcatLayerChiselMixin, ConcatLayer2DMixin):
    register: ClassVar[bool] = True

