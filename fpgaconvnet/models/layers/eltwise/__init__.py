from typing import ClassVar
from dataclasses import dataclass

from fpgaconvnet.models.layers import LayerBase
from .base import EltwiseLayerBase, EltwiseLayerChiselMixin, EltwiseLayer2DMixin

@dataclass(kw_only=True)
class EltwiseLayerChisel2D(EltwiseLayerChiselMixin, EltwiseLayer2DMixin):
    register: ClassVar[bool] = True

