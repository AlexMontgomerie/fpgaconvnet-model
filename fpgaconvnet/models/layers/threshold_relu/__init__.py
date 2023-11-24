from typing import ClassVar
from dataclasses import dataclass

from fpgaconvnet.models.layers import LayerBase
from fpgaconvnet.models.layers.traits import Layer2D, Layer3D
from .base import ThresholdReLULayerBase, ThresholdReLULayerChiselMixin, ThresholdReLULayerHLSMixin

@dataclass(kw_only=True)
class ThresholdReLULayerChisel2D(ThresholdReLULayerChiselMixin, Layer2D, ThresholdReLULayerBase):
    register: ClassVar[bool] = True

@dataclass(kw_only=True)
class ThresholdReLULayerChisel3D(ThresholdReLULayerChiselMixin, Layer3D, ThresholdReLULayerBase):
    register: ClassVar[bool] = True

@dataclass(kw_only=True)
class ThresholdReLULayerHLS2D(ThresholdReLULayerHLSMixin, Layer2D, ThresholdReLULayerBase):
    register: ClassVar[bool] = True

@dataclass(kw_only=True)
class ThresholdReLULayerHLS3D(ThresholdReLULayerHLSMixin, Layer3D, ThresholdReLULayerBase):
    register: ClassVar[bool] = True
