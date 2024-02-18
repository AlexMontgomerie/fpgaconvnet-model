# import math
# from typing import ClassVar, Any
# from dataclasses import dataclass
# from collections import OrderedDict

# import numpy as np
# from dacite import from_dict
# import networkx as nx

# import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
# from fpgaconvnet.models.layers.utils import get_factors
# from fpgaconvnet.data_types import FixedPoint
# from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

# from fpgaconvnet.models.layers import LayerBase
# from fpgaconvnet.models.layers.traits import LayerMatchingCoarse
# from fpgaconvnet.models.modules import ModuleBase

# from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY
# from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

# @dataclass(kw_only=True)
# class ChopLayerBase(LayerMatchingCoarse, LayerBase):

#     ports: int
#     split: list[int]
#     data_t: FixedPoint = FixedPoint(16,8)

#     name: ClassVar[str] = "chop"

#     def __setattr__(self, name: str, value: Any) -> None:

#         if not hasattr(self, "is_init"):
#             super().__setattr__(name, value)
#             return

#         try:
#             match name:
#                 case "ports" | "ports_out":
#                     super().__setattr__("ports", value)
#                     super().__setattr__("ports_out", value)

#                 case "ports_in":
#                     raise ValueError("ERROR: cannot set ports_in (always 1)")

#                 case _:
#                     super().__setattr__(name, value)

#         except AttributeError:
#             print(f"WARNING: unable to set attribute {name}, trying super method")
#             super().__setattr__(name, value)

# class ChopLayerChiselMixin(ChopLayerBase):

#     backend: ClassVar[BACKEND] = BACKEND.CHISEL

#     @property
#     def module_lookup(self) -> dict:
#         return OrderedDict({
#             "chop": self.get_chop_parameters,
#         })

#     def get_chop_parameters(self) -> dict:
#         return {
#             "repetitions": math.prod(self.input_shape())//self.streams(),
#             "streams": self.coarse,
#             "data_t": self.data_t,
#         }

#     def build_module_graph(self) -> nx.DiGraph:

#         # get the module graph
#         self.graph = nx.DiGraph()

#         # add the chop module
#         self.graph.add_node("chop", module=self.modules["chop"])


# class ChopLayerHLSMixin(ChopLayerBase):

#     backend: ClassVar[BACKEND] = BACKEND.HLS

#     @property
#     def module_lookup(self) -> dict:
#         return OrderedDict({
#             "chop": self.get_chop_parameters,
#         })

#     def get_chop_parameters(self) -> dict:
#         return {
#             **self.input_shape_dict(),
#             "channels": self.channels//self.coarse,
#             "data_t": self.data_t,
#         }

#     def build_module_graph(self) -> nx.DiGraph:

#         # get the module graph
#         self.graph = nx.DiGraph()

#         # add the chop module
#         for i in range(self.coarse):
#             self.graph.add_node(f"chop_{i}", module=self.modules["chop"])

