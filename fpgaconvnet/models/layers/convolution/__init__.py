from .base import ConvolutionLayerBaseMeta, ConvolutionLayerBase, ConvolutionLayerTrait2D
from .backend import ConvolutionLayerTraitChisel

from dataclasses import dataclass, field

@dataclass(kw_only=True)
class ConvolutionLayerChisel(ConvolutionLayerTraitChisel, ConvolutionLayerTrait2D, ConvolutionLayerBase):
    pass

