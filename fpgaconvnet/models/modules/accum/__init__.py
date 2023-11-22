from typing import Optional

from .chisel import AccumChisel
from .hls import AccumHLS, AccumHLS3D

from fpgaconvnet.models.modules import CHISEL_RSC_TYPES
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model

try:
    DEFAULT_ACCUM_RSC_MODELS = { rsc_type: get_cached_resource_model(AccumChisel,
                                    rsc_type, "default") for rsc_type in CHISEL_RSC_TYPES }
except FileNotFoundError:
    print("CRITICAL WARNING: default resource models not found, default resource modelling will fail")

@eval_resource_model.register
def eval_resource_model(m: AccumChisel, rsc_type: str, model: Optional[ResourceModel] = None) -> int:

    # get the resource model
    model = model if model is not None else DEFAULT_ACCUM_RSC_MODELS[rsc_type]

    # check the correct resource type
    assert rsc_type in CHISEL_RSC_TYPES, f"Invalid resource type: {rsc_type}"
    assert rsc_type == model.rsc_type, f"Incompatible resource type with model: {rsc_type}"

    # get the resource model
    match rsc_type:
        case "DSP":
            return 0
        case _:
            return model(m)


