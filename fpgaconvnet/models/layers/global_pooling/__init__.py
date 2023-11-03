from .base import GlobalPoolingLayerBaseMeta, GlobalPoolingLayerBase, GlobalPoolingLayerTrait2D
from .backend import GlobalPoolingLayerTraitChisel

from dataclasses import dataclass, field

@dataclass(kw_only=True)
class GlobalPoolingLayerChisel(GlobalPoolingLayerTraitChisel,
        GlobalPoolingLayerTrait2D, GlobalPoolingLayerBase):
    pass

