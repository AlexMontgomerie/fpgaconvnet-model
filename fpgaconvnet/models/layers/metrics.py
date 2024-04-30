from .layer import LayerBase
from fpgaconvnet.platform import PlatformBase
from fpgaconvnet.models.modules.metrics import get_module_resources

def get_layer_resources(layer: LayerBase, rsc_type: str, platform: PlatformBase) -> float:
    """
    Get the resource usage of a given layer instance, for the particular
    platform architecture.

    Args:
        layer: the layer to evaluate
        rsc_type: the resource type to evaluate
        platform: the platform to evaluate the layer on

    Returns:
        dict[str,int]: the amount of resources
    """

    # initialise the resources as zero
    resources = 0

     # iterate over the nodes of the graph
    for node in layer.module_graph.nodes:

        # get the module
        module = layer.module_graph.nodes[node]["module"]

        # get the resource model from the platform
        resources += get_module_resources(module, rsc_type, platform)

    # return the resources
    return resources

def get_layer_resources_all(layer: LayerBase, platform: PlatformBase) -> dict[str,float]:
    # iterate over the resource types of the platform, and get the resources
    return { rsc_type: get_layer_resources(layer, rsc_type, platform) for rsc_type in platform.resource_types }


def get_layer_latency(layer: LayerBase, platform: PlatformBase):
    pass

def get_layer_pipeline_delay(layer: LayerBase, platform: PlatformBase):
    pass

def get_layer_throughput(layer: LayerBase, platform: PlatformBase):
    pass

def get_layer_frequency(layer: LayerBase, platform: PlatformBase):
    pass

def get_layer_power(layer: LayerBase, platform: PlatformBase):
    pass

