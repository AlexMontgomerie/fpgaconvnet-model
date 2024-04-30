import os
import toml
from typing import ClassVar
from dataclasses import dataclass

from .platform import PlatformBase
from .amd import ZynqPlatform, ZynqUltrascalePlatform


def get_fpga_part_family(part: str) -> str: return part[:4]

def build_platform_from_toml(platform_path: str):

    # make sure toml configuration
    assert os.path.splitext(platform_path)[1] == ".toml", "must be a TOML configuration file"

    # parse platform configuration toml file
    with open(platform_path, "r") as f:
        conf = toml.load(f)

    # find and initialise the platform
    family = get_fpga_part_family(conf["device"]["part"])
    match family:
        case ZynqPlatform.family:
            return ZynqPlatform.from_toml(platform_path)
        case ZynqUltrascalePlatform.family:
            return ZynqUltrascalePlatform.from_toml(platform_path)
        case _:
            raise NotImplementedError(f"platform {family} not supported")


# # initialise some platforms
# DEFAULT_PLATFORMS = {
#     "zedboard": build_platform_from_toml(f"{os.path.dirname(__file__)}/configs/zedboard.toml"),
#     # "zc706": build_platform_from_toml(f"{os.path.dirname(__file__)}/configs/zc706.toml"),
#     "zcu104": build_platform_from_toml(f"{os.path.dirname(__file__)}/configs/zcu104.toml"),
# }

DEFAULT_HLS_PLATFORM = build_platform_from_toml(f"{os.path.dirname(__file__)}/configs/zc706.toml")
DEFAULT_CHISEL_PLATFORM = build_platform_from_toml(f"{os.path.dirname(__file__)}/configs/zcu104.toml")
