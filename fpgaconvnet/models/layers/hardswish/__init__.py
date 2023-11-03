from .base import HardswishLayerBaseMeta, HardswishLayerBase, HardswishLayerTrait2D
from .backend import HardswishLayerTraitChisel

from dataclasses import dataclass, field

@dataclass(kw_only=True)
class HardswishLayerChisel(HardswishLayerTraitChisel,
        HardswishLayerTrait2D, HardswishLayerBase):
    pass

