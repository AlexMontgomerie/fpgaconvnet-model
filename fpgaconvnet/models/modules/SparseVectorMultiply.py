# import json
# from dataclasses import dataclass
# from typing import ClassVar

# import numpy as np

# from fpgaconvnet.models.modules import SparseVectorMultiply
# from fpgaconvnet.chisel import Hardware

# from fpgaconvnet.chisel.tools.generate_data import generate_data_fixed_point

# @dataclass(kw_only=True)
# class SparseVectorMultiplyHardware(Hardware, SparseVectorMultiply):
#     hierachy:   ClassVar[str] = "module"
#     hw_type:    ClassVar[str] = "sparse_vector_multiply"
#     vsource:    ClassVar[str] = "SparseVectorMultiplyFixed.v"
#     top_module: ClassVar[str] = "SparseVectorMultiplyFixed"
#     module:     ClassVar[str] = "tests.modules.test_sparse_vector_multiply"

#     @property
#     def input_shape(self):
#         return [ self.streams ]

#     def generate_test_data(self, target_dir=None, save=False):

#         # inherit target directory if argument is "None"
#         if target_dir == None:
#             target_dir = self.target_dir

#         # data in
#         data_in = generate_data_fixed_point(
#                 self.input_shape,
#                 data_width=self.data_width,
#                 binary_point=self.data_width//2
#             )

#         # weights
#         weights = generate_data_fixed_point(
#                 [ self.streams ],
#                 data_width=self.data_width,
#                 binary_point=self.data_width//2
#             )

#         # data out
#         data_out = self.functional_model(data_in, weights)

#         if save:

#             # save input as .dat files
#             with open(f"{target_dir}/input.dat", 'w') as f:
#                 f.write("\n".join([ str(i) for i in data_in.reshape(-1).tolist() ]))

#             # save weights as .dat files
#             with open(f"{target_dir}/weights.dat", 'w') as f:
#                 f.write("\n".join([ str(i) for i in weights.reshape(-1).tolist() ]))

#             # save output as .dat files
#             with open(f"{target_dir}/output.dat", 'w') as f:
#                 f.write("\n".join([ str(i) for i in data_out.reshape(-1).tolist() ]))

#         # return the test data
#         return data_in, weights, data_out


