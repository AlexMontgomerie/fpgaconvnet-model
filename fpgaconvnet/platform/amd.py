from dataclasses import dataclass
from typing import ClassVar

from .platform import PlatformBase

@dataclass
class ZynqPlatform(PlatformBase):
    family: ClassVar[str] = "xc7z"
    resource_types: ClassVar[list[str]] = ["BRAM", "DSP", "FF", "LUT"]

@dataclass
class ZynqUltrascalePlatform(PlatformBase):
    family: ClassVar[str] = "xczu"
    resource_types: ClassVar[list[str]] = ["BRAM", "DSP", "FF", "LUT", "URAM"]
    # resource_types: ClassVar[list[str]] = ["BRAM", "DSP", "FF", "LUT", "URAM"]


