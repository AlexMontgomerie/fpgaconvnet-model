from fpgaconvnet.platform import PlatformBase
from fpgaconvnet.models.modules import ModuleBase
from fpgaconvnet.models.modules.resources import eval_resource_model

def get_module_resources(module: ModuleBase, rsc_type: str, platform: PlatformBase) -> float:
    """
    Get the resource usage of a given module instance, for the particular
    platform architecture.

    Args:
        module: the module to evaluate
        rsc_type: the resource type to evaluate
        platform: the platform to evaluate the module on

    Returns:
        float: the amount of resources
    """
    # get the resource model from the platform
    rsc_model = platform.get_resource_model(module, rsc_type)

    # evaluate the resource model
    return eval_resource_model(module, rsc_type, rsc_model)

def get_module_resources_all(module: ModuleBase, platform: PlatformBase) -> dict[str,float]:

    # iterate over the resource types of the platform , and get the resources
    return { rsc_type: get_module_resources(module, rsc_type, platform) for rsc_type in platform.resource_types }

def get_module_latency(module: ModuleBase, platform: PlatformBase):
    pass

def get_module_throughput(module: ModuleBase, platform: PlatformBase):
    pass

def get_module_max_frequency(module: ModuleBase, platform: PlatformBase):
    # TODO: get an estimate of the module max frequency
    return platform.freq

def get_module_power(module: ModuleBase, platform: PlatformBase):
    pass

