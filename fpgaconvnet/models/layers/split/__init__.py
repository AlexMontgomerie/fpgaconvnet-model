from typing import ClassVar
from dataclasses import dataclass

from fpgaconvnet.models.layers import LayerBase
from .base import SplitLayerBase, SplitLayerChiselMixin, SplitLayer2DMixin

@dataclass(kw_only=True)
class SplitLayerChisel2D(SplitLayerChiselMixin, SplitLayer2DMixin):
    register: ClassVar[bool] = True

