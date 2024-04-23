from fpgaconvnet.platform import PlatformBase
from fpgaconvnet.models.modules import ModuleBase
from fpgaconvnet.models.modules.resources import eval_resource_model

def get_module_resources(module: ModuleBase, rsc_type: str, platform: PlatformBase):

    # get the resource model from the platform
    rsc_model = platform.get_resource_model(module, rsc_type)

    # evaluate the resource model
    return eval_resource_model(module, rsc_type, rsc_model)

def get_module_latency(module: ModuleBase, platform: PlatformBase):
    pass

def get_module_throughput(module: ModuleBase, platform: PlatformBase):
    pass

def get_module_max_frequency(module: ModuleBase, platform: PlatformBase):
    # TODO: get an estimate of the module max frequency
    return platform.board_freq

def get_module_power(module: ModuleBase, platform: PlatformBase):
    pass

