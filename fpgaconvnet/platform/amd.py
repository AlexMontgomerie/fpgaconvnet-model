from dataclasses import dataclass
from typing import ClassVar

from .platform import PlatformBase

# yosys document well the different types of AMD family (https://github.com/YosysHQ/yosys/blob/1a54e8d47b8cef93d51487cc5ffc092fc9295bcd/techlibs/xilinx/synth_xilinx.cc#L31)

@dataclass
class ZynqPlatform(PlatformBase):
    family: ClassVar[str] = "xc7z"
    resource_types: ClassVar[list[str]] = ["BRAM", "DSP", "FF", "LUT"]

@dataclass
class ZynqUltrascalePlatform(PlatformBase):
    family: ClassVar[str] = "xczu"
    resource_types: ClassVar[list[str]] = ["BRAM", "DSP", "FF", "LUT", "URAM"]

@dataclass
class ZynqUltrascalePlusPlatform(PlatformBase):
    family: ClassVar[str] = "xczup"
    resource_types: ClassVar[list[str]] = ["BRAM", "DSP", "FF", "LUT", "URAM"]

@dataclass
class UltrascalePlatform(PlatformBase):
    family: ClassVar[str] = "xcu"
    resource_types: ClassVar[list[str]] = ["BRAM", "DSP", "FF", "LUT", "URAM"]

@dataclass
class UltrascalePlusPlatform(PlatformBase):
    family: ClassVar[str] = "xcup"
    resource_types: ClassVar[list[str]] = ["BRAM", "DSP", "FF", "LUT", "URAM"]

@dataclass
class Series7Platform(PlatformBase):
    family: ClassVar[str] = "xc7"
    resource_types: ClassVar[list[str]] = ["BRAM", "DSP", "FF", "LUT"]

@dataclass
class Spartan6Platform(PlatformBase):
    family: ClassVar[str] = "xc6s"
    resource_types: ClassVar[list[str]] = ["BRAM", "DSP", "FF", "LUT"]

@dataclass
class Virtex6Platform(PlatformBase):
    family: ClassVar[str] = "xc5v"
    resource_types: ClassVar[list[str]] = ["BRAM", "DSP", "FF", "LUT"]

