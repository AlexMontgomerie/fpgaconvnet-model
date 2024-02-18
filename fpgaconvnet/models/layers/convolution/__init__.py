from typing import ClassVar
from dataclasses import dataclass

from .base import ConvolutionLayerBase, ConvolutionLayer2DMixin, ConvolutionLayer3DMixin
from .backend import ConvolutionLayerChiselMixin, ConvolutionLayerHLSMixin

@dataclass(kw_only=True)
class ConvolutionLayerChisel2D(ConvolutionLayerChiselMixin, ConvolutionLayer2DMixin):
    register: ClassVar[bool] = True

@dataclass(kw_only=True)
class ConvolutionLayerChisel3D(ConvolutionLayerChiselMixin, ConvolutionLayer3DMixin):
    register: ClassVar[bool] = True

@dataclass(kw_only=True)
class ConvolutionLayerHLS2D(ConvolutionLayerHLSMixin, ConvolutionLayer2DMixin):
    register: ClassVar[bool] = True

@dataclass(kw_only=True)
class ConvolutionLayerHLS3D(ConvolutionLayerHLSMixin, ConvolutionLayer3DMixin):
    register: ClassVar[bool] = True

