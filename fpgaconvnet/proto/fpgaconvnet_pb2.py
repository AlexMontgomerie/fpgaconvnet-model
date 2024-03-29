# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: fpgaconvnet.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='fpgaconvnet.proto',
  package='',
  syntax='proto2',
  serialized_pb=_b('\n\x11\x66pgaconvnet.proto\"&\n\x06stream\x12\x0c\n\x04name\x18\x01 \x02(\t\x12\x0e\n\x06\x63oarse\x18\x02 \x02(\x05\"\xa4\x04\n\tparameter\x12\x17\n\x0c\x62uffer_depth\x18\x01 \x02(\x05:\x01\x32\x12\x12\n\nbatch_size\x18\x02 \x02(\x05\x12\x0f\n\x07rows_in\x18\x03 \x02(\x05\x12\x0f\n\x07\x63ols_in\x18\x04 \x02(\x05\x12\x13\n\x0b\x63hannels_in\x18\x05 \x02(\x05\x12\x10\n\x08rows_out\x18\x06 \x02(\x05\x12\x10\n\x08\x63ols_out\x18\x07 \x02(\x05\x12\x14\n\x0c\x63hannels_out\x18\x08 \x02(\x05\x12\x11\n\tcoarse_in\x18\t \x02(\x05\x12\x12\n\ncoarse_out\x18\n \x02(\x05\x12\x0e\n\x06\x63oarse\x18\x0b \x01(\x05\x12\x14\n\x0c\x63oarse_group\x18\x0c \x01(\x05\x12\x0e\n\x06groups\x18\r \x01(\x05\x12\x0c\n\x04\x66ine\x18\x0e \x01(\x05\x12\x0f\n\x07\x66ilters\x18\x0f \x01(\x05\x12\x0f\n\x07pad_top\x18\x10 \x01(\x05\x12\x11\n\tpad_right\x18\x11 \x01(\x05\x12\x10\n\x08pad_left\x18\x12 \x01(\x05\x12\x12\n\npad_bottom\x18\x13 \x01(\x05\x12\x13\n\x0bkernel_size\x18\x14 \x03(\x05\x12\x0e\n\x06stride\x18\x15 \x03(\x05\x12\x12\n\ndata_width\x18\x16 \x01(\x05\x12\x14\n\x0cweight_width\x18\x17 \x01(\x05\x12\x11\n\tacc_width\x18\x18 \x01(\x05\x12\x13\n\x0binput_width\x18\x19 \x01(\x05\x12\x14\n\x0coutput_width\x18\x1a \x01(\x05\x12\x10\n\x08has_bias\x18\x1b \x01(\x05\x12\x14\n\x0c\x62iases_width\x18\x1c \x01(\x05\"\xf0\x02\n\x05layer\x12\x0c\n\x04name\x18\x01 \x02(\t\x12\x1b\n\nstreams_in\x18\x02 \x03(\x0b\x32\x07.stream\x12\x1c\n\x0bstreams_out\x18\x03 \x03(\x0b\x32\x07.stream\x12\x1e\n\nparameters\x18\x04 \x02(\x0b\x32\n.parameter\x12\x0f\n\x07node_in\x18\x05 \x02(\t\x12\x10\n\x08node_out\x18\x06 \x02(\t\x12\x1f\n\x04type\x18\x07 \x02(\x0e\x32\x11.layer.layer_type\x12\x14\n\x0cweights_path\x18\x08 \x01(\t\x12\x11\n\tbias_path\x18\t \x01(\t\"\x90\x01\n\nlayer_type\x12\x0f\n\x0b\x43ONVOLUTION\x10\x00\x12\x0b\n\x07POOLING\x10\x01\x12\x08\n\x04RELU\x10\x02\x12\x0b\n\x07SQUEEZE\x10\x03\x12\x11\n\rINNER_PRODUCT\x10\x04\x12\n\n\x06\x43ONCAT\x10\x05\x12\x0e\n\nBATCH_NORM\x10\x06\x12\t\n\x05SPLIT\x10\x07\x12\x13\n\x0f\x41VERAGE_POOLING\x10\x08\"\xbe\x01\n\tpartition\x12\n\n\x02id\x18\x01 \x02(\x05\x12\r\n\x05ports\x18\x02 \x02(\x05\x12\x12\n\nbatch_size\x18\x03 \x02(\x05\x12\x12\n\ninput_node\x18\x04 \x02(\t\x12\x13\n\x0boutput_node\x18\x05 \x02(\t\x12 \n\x18weights_reloading_factor\x18\x06 \x02(\x05\x12\x1f\n\x17weights_reloading_layer\x18\x07 \x02(\t\x12\x16\n\x06layers\x18\x08 \x03(\x0b\x32\x06.layer\"+\n\npartitions\x12\x1d\n\tpartition\x18\x01 \x03(\x0b\x32\n.partition')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)



_LAYER_LAYER_TYPE = _descriptor.EnumDescriptor(
  name='layer_type',
  full_name='layer.layer_type',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='CONVOLUTION', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='POOLING', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RELU', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SQUEEZE', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INNER_PRODUCT', index=4, number=4,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CONCAT', index=5, number=5,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BATCH_NORM', index=6, number=6,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SPLIT', index=7, number=7,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AVERAGE_POOLING', index=8, number=8,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=837,
  serialized_end=981,
)
_sym_db.RegisterEnumDescriptor(_LAYER_LAYER_TYPE)


_STREAM = _descriptor.Descriptor(
  name='stream',
  full_name='stream',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='stream.name', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='coarse', full_name='stream.coarse', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=21,
  serialized_end=59,
)


_PARAMETER = _descriptor.Descriptor(
  name='parameter',
  full_name='parameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='buffer_depth', full_name='parameter.buffer_depth', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=True, default_value=2,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='batch_size', full_name='parameter.batch_size', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='rows_in', full_name='parameter.rows_in', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='cols_in', full_name='parameter.cols_in', index=3,
      number=4, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='channels_in', full_name='parameter.channels_in', index=4,
      number=5, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='rows_out', full_name='parameter.rows_out', index=5,
      number=6, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='cols_out', full_name='parameter.cols_out', index=6,
      number=7, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='channels_out', full_name='parameter.channels_out', index=7,
      number=8, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='coarse_in', full_name='parameter.coarse_in', index=8,
      number=9, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='coarse_out', full_name='parameter.coarse_out', index=9,
      number=10, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='coarse', full_name='parameter.coarse', index=10,
      number=11, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='coarse_group', full_name='parameter.coarse_group', index=11,
      number=12, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='groups', full_name='parameter.groups', index=12,
      number=13, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='fine', full_name='parameter.fine', index=13,
      number=14, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='filters', full_name='parameter.filters', index=14,
      number=15, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pad_top', full_name='parameter.pad_top', index=15,
      number=16, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pad_right', full_name='parameter.pad_right', index=16,
      number=17, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pad_left', full_name='parameter.pad_left', index=17,
      number=18, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pad_bottom', full_name='parameter.pad_bottom', index=18,
      number=19, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='kernel_size', full_name='parameter.kernel_size', index=19,
      number=20, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='stride', full_name='parameter.stride', index=20,
      number=21, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='data_width', full_name='parameter.data_width', index=21,
      number=22, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='weight_width', full_name='parameter.weight_width', index=22,
      number=23, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='acc_width', full_name='parameter.acc_width', index=23,
      number=24, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='input_width', full_name='parameter.input_width', index=24,
      number=25, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='output_width', full_name='parameter.output_width', index=25,
      number=26, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='has_bias', full_name='parameter.has_bias', index=26,
      number=27, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='biases_width', full_name='parameter.biases_width', index=27,
      number=28, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=62,
  serialized_end=610,
)


_LAYER = _descriptor.Descriptor(
  name='layer',
  full_name='layer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='layer.name', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='streams_in', full_name='layer.streams_in', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='streams_out', full_name='layer.streams_out', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='parameters', full_name='layer.parameters', index=3,
      number=4, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='node_in', full_name='layer.node_in', index=4,
      number=5, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='node_out', full_name='layer.node_out', index=5,
      number=6, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='type', full_name='layer.type', index=6,
      number=7, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='weights_path', full_name='layer.weights_path', index=7,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='bias_path', full_name='layer.bias_path', index=8,
      number=9, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _LAYER_LAYER_TYPE,
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=613,
  serialized_end=981,
)


_PARTITION = _descriptor.Descriptor(
  name='partition',
  full_name='partition',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='partition.id', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ports', full_name='partition.ports', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='batch_size', full_name='partition.batch_size', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='input_node', full_name='partition.input_node', index=3,
      number=4, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='output_node', full_name='partition.output_node', index=4,
      number=5, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='weights_reloading_factor', full_name='partition.weights_reloading_factor', index=5,
      number=6, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='weights_reloading_layer', full_name='partition.weights_reloading_layer', index=6,
      number=7, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='layers', full_name='partition.layers', index=7,
      number=8, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=984,
  serialized_end=1174,
)


_PARTITIONS = _descriptor.Descriptor(
  name='partitions',
  full_name='partitions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='partition', full_name='partitions.partition', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1176,
  serialized_end=1219,
)

_LAYER.fields_by_name['streams_in'].message_type = _STREAM
_LAYER.fields_by_name['streams_out'].message_type = _STREAM
_LAYER.fields_by_name['parameters'].message_type = _PARAMETER
_LAYER.fields_by_name['type'].enum_type = _LAYER_LAYER_TYPE
_LAYER_LAYER_TYPE.containing_type = _LAYER
_PARTITION.fields_by_name['layers'].message_type = _LAYER
_PARTITIONS.fields_by_name['partition'].message_type = _PARTITION
DESCRIPTOR.message_types_by_name['stream'] = _STREAM
DESCRIPTOR.message_types_by_name['parameter'] = _PARAMETER
DESCRIPTOR.message_types_by_name['layer'] = _LAYER
DESCRIPTOR.message_types_by_name['partition'] = _PARTITION
DESCRIPTOR.message_types_by_name['partitions'] = _PARTITIONS

stream = _reflection.GeneratedProtocolMessageType('stream', (_message.Message,), dict(
  DESCRIPTOR = _STREAM,
  __module__ = 'fpgaconvnet_pb2'
  # @@protoc_insertion_point(class_scope:stream)
  ))
_sym_db.RegisterMessage(stream)

parameter = _reflection.GeneratedProtocolMessageType('parameter', (_message.Message,), dict(
  DESCRIPTOR = _PARAMETER,
  __module__ = 'fpgaconvnet_pb2'
  # @@protoc_insertion_point(class_scope:parameter)
  ))
_sym_db.RegisterMessage(parameter)

layer = _reflection.GeneratedProtocolMessageType('layer', (_message.Message,), dict(
  DESCRIPTOR = _LAYER,
  __module__ = 'fpgaconvnet_pb2'
  # @@protoc_insertion_point(class_scope:layer)
  ))
_sym_db.RegisterMessage(layer)

partition = _reflection.GeneratedProtocolMessageType('partition', (_message.Message,), dict(
  DESCRIPTOR = _PARTITION,
  __module__ = 'fpgaconvnet_pb2'
  # @@protoc_insertion_point(class_scope:partition)
  ))
_sym_db.RegisterMessage(partition)

partitions = _reflection.GeneratedProtocolMessageType('partitions', (_message.Message,), dict(
  DESCRIPTOR = _PARTITIONS,
  __module__ = 'fpgaconvnet_pb2'
  # @@protoc_insertion_point(class_scope:partitions)
  ))
_sym_db.RegisterMessage(partitions)


# @@protoc_insertion_point(module_scope)
