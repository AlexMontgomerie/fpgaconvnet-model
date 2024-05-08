import os
import glob

from fpgaconvnet.platform import build_platform_from_toml

# get all the platform configuration files
PLATFORM_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "fpgaconvnet", "platform", "configs")
platform_configs = glob.glob(os.path.join(PLATFORM_CONFIG_DIR, "*.toml"))

# iterate over all the platform configuration files
for platform_path in platform_configs:

    # load the platform configuration
    try:
        platform = build_platform_from_toml(platform_path)
    except Exception as e:
        print(f"Error loading platform configuration {platform_path}: {e}")
        continue

    # build all the resource models for the platform
    try:
        print(f"Building resource models for platform {platform.part}")
        platform.build_all_resource_models()
    except Exception as e:
        print(f"Error building resource models for platform {platform.part}: {e}")
