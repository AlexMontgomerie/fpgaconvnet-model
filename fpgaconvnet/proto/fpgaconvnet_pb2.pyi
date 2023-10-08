from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

ACTIVATION: layer_type
AVERAGE_POOLING: layer_type
BATCH_NORM: layer_type
CHOP: layer_type
CONCAT: layer_type
CONVOLUTION: layer_type
DESCRIPTOR: _descriptor.FileDescriptor
ELTWISE: layer_type
INNER_PRODUCT: layer_type
POOLING: layer_type
RESIZE: layer_type
SPLIT: layer_type
SQUEEZE: layer_type

class fixed_point(_message.Message):
    __slots__ = ["binary_point", "width"]
    BINARY_POINT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    binary_point: int
    width: int
    def __init__(self, width: _Optional[int] = ..., binary_point: _Optional[int] = ...) -> None: ...

class layer(_message.Message):
    __slots__ = ["bias_path", "name", "onnx_node", "op_type", "parameters", "streams_in", "streams_out", "type", "weights_path"]
    BIAS_PATH_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ONNX_NODE_FIELD_NUMBER: _ClassVar[int]
    OP_TYPE_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    STREAMS_IN_FIELD_NUMBER: _ClassVar[int]
    STREAMS_OUT_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_PATH_FIELD_NUMBER: _ClassVar[int]
    bias_path: str
    name: str
    onnx_node: str
    op_type: str
    parameters: parameter
    streams_in: _containers.RepeatedCompositeFieldContainer[stream]
    streams_out: _containers.RepeatedCompositeFieldContainer[stream]
    type: layer_type
    weights_path: str
    def __init__(self, name: _Optional[str] = ..., onnx_node: _Optional[str] = ..., type: _Optional[_Union[layer_type, str]] = ..., op_type: _Optional[str] = ..., streams_in: _Optional[_Iterable[_Union[stream, _Mapping]]] = ..., streams_out: _Optional[_Iterable[_Union[stream, _Mapping]]] = ..., weights_path: _Optional[str] = ..., bias_path: _Optional[str] = ..., parameters: _Optional[_Union[parameter, _Mapping]] = ...) -> None: ...

class parameter(_message.Message):
    __slots__ = ["acc_t", "batch_size", "bias_quant", "bias_t", "block_floating_point", "channels_in", "channels_in_array", "channels_out", "channels_out_array", "coarse", "coarse_group", "coarse_in", "coarse_out", "cols_in", "cols_in_array", "cols_out", "cols_out_array", "data_t", "depth_in", "depth_in_array", "depth_out", "depth_out_array", "filters", "fine", "groups", "has_bias", "input_quant", "input_t", "kernel_cols", "kernel_depth", "kernel_rows", "kernel_size", "mem_bw_in", "mem_bw_in_array", "mem_bw_out", "mem_bw_out_array", "off_chip_buffer_size", "off_chip_interval", "on_chip_addr_range", "output_quant", "output_t", "pad_back", "pad_bottom", "pad_front", "pad_left", "pad_right", "pad_top", "ports_in", "ports_out", "rows_in", "rows_in_array", "rows_out", "rows_out_array", "sparsity", "split", "stream_inputs", "stream_outputs", "stream_weights", "stride", "stride_cols", "stride_depth", "stride_rows", "use_uram", "weight_quant", "weight_t"]
    ACC_T_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    BIAS_QUANT_FIELD_NUMBER: _ClassVar[int]
    BIAS_T_FIELD_NUMBER: _ClassVar[int]
    BLOCK_FLOATING_POINT_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_IN_ARRAY_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_IN_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_OUT_ARRAY_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_OUT_FIELD_NUMBER: _ClassVar[int]
    COARSE_FIELD_NUMBER: _ClassVar[int]
    COARSE_GROUP_FIELD_NUMBER: _ClassVar[int]
    COARSE_IN_FIELD_NUMBER: _ClassVar[int]
    COARSE_OUT_FIELD_NUMBER: _ClassVar[int]
    COLS_IN_ARRAY_FIELD_NUMBER: _ClassVar[int]
    COLS_IN_FIELD_NUMBER: _ClassVar[int]
    COLS_OUT_ARRAY_FIELD_NUMBER: _ClassVar[int]
    COLS_OUT_FIELD_NUMBER: _ClassVar[int]
    DATA_T_FIELD_NUMBER: _ClassVar[int]
    DEPTH_IN_ARRAY_FIELD_NUMBER: _ClassVar[int]
    DEPTH_IN_FIELD_NUMBER: _ClassVar[int]
    DEPTH_OUT_ARRAY_FIELD_NUMBER: _ClassVar[int]
    DEPTH_OUT_FIELD_NUMBER: _ClassVar[int]
    FILTERS_FIELD_NUMBER: _ClassVar[int]
    FINE_FIELD_NUMBER: _ClassVar[int]
    GROUPS_FIELD_NUMBER: _ClassVar[int]
    HAS_BIAS_FIELD_NUMBER: _ClassVar[int]
    INPUT_QUANT_FIELD_NUMBER: _ClassVar[int]
    INPUT_T_FIELD_NUMBER: _ClassVar[int]
    KERNEL_COLS_FIELD_NUMBER: _ClassVar[int]
    KERNEL_DEPTH_FIELD_NUMBER: _ClassVar[int]
    KERNEL_ROWS_FIELD_NUMBER: _ClassVar[int]
    KERNEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    MEM_BW_IN_ARRAY_FIELD_NUMBER: _ClassVar[int]
    MEM_BW_IN_FIELD_NUMBER: _ClassVar[int]
    MEM_BW_OUT_ARRAY_FIELD_NUMBER: _ClassVar[int]
    MEM_BW_OUT_FIELD_NUMBER: _ClassVar[int]
    OFF_CHIP_BUFFER_SIZE_FIELD_NUMBER: _ClassVar[int]
    OFF_CHIP_INTERVAL_FIELD_NUMBER: _ClassVar[int]
    ON_CHIP_ADDR_RANGE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_QUANT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_T_FIELD_NUMBER: _ClassVar[int]
    PAD_BACK_FIELD_NUMBER: _ClassVar[int]
    PAD_BOTTOM_FIELD_NUMBER: _ClassVar[int]
    PAD_FRONT_FIELD_NUMBER: _ClassVar[int]
    PAD_LEFT_FIELD_NUMBER: _ClassVar[int]
    PAD_RIGHT_FIELD_NUMBER: _ClassVar[int]
    PAD_TOP_FIELD_NUMBER: _ClassVar[int]
    PORTS_IN_FIELD_NUMBER: _ClassVar[int]
    PORTS_OUT_FIELD_NUMBER: _ClassVar[int]
    ROWS_IN_ARRAY_FIELD_NUMBER: _ClassVar[int]
    ROWS_IN_FIELD_NUMBER: _ClassVar[int]
    ROWS_OUT_ARRAY_FIELD_NUMBER: _ClassVar[int]
    ROWS_OUT_FIELD_NUMBER: _ClassVar[int]
    SPARSITY_FIELD_NUMBER: _ClassVar[int]
    SPLIT_FIELD_NUMBER: _ClassVar[int]
    STREAM_INPUTS_FIELD_NUMBER: _ClassVar[int]
    STREAM_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    STREAM_WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    STRIDE_COLS_FIELD_NUMBER: _ClassVar[int]
    STRIDE_DEPTH_FIELD_NUMBER: _ClassVar[int]
    STRIDE_FIELD_NUMBER: _ClassVar[int]
    STRIDE_ROWS_FIELD_NUMBER: _ClassVar[int]
    USE_URAM_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_QUANT_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_T_FIELD_NUMBER: _ClassVar[int]
    acc_t: fixed_point
    batch_size: int
    bias_quant: quant_format
    bias_t: fixed_point
    block_floating_point: bool
    channels_in: int
    channels_in_array: _containers.RepeatedScalarFieldContainer[int]
    channels_out: int
    channels_out_array: _containers.RepeatedScalarFieldContainer[int]
    coarse: int
    coarse_group: int
    coarse_in: int
    coarse_out: int
    cols_in: int
    cols_in_array: _containers.RepeatedScalarFieldContainer[int]
    cols_out: int
    cols_out_array: _containers.RepeatedScalarFieldContainer[int]
    data_t: fixed_point
    depth_in: int
    depth_in_array: _containers.RepeatedScalarFieldContainer[int]
    depth_out: int
    depth_out_array: _containers.RepeatedScalarFieldContainer[int]
    filters: int
    fine: int
    groups: int
    has_bias: bool
    input_quant: quant_format
    input_t: fixed_point
    kernel_cols: int
    kernel_depth: int
    kernel_rows: int
    kernel_size: _containers.RepeatedScalarFieldContainer[int]
    mem_bw_in: float
    mem_bw_in_array: _containers.RepeatedScalarFieldContainer[float]
    mem_bw_out: float
    mem_bw_out_array: _containers.RepeatedScalarFieldContainer[float]
    off_chip_buffer_size: int
    off_chip_interval: int
    on_chip_addr_range: int
    output_quant: quant_format
    output_t: fixed_point
    pad_back: int
    pad_bottom: int
    pad_front: int
    pad_left: int
    pad_right: int
    pad_top: int
    ports_in: int
    ports_out: int
    rows_in: int
    rows_in_array: _containers.RepeatedScalarFieldContainer[int]
    rows_out: int
    rows_out_array: _containers.RepeatedScalarFieldContainer[int]
    sparsity: float
    split: _containers.RepeatedScalarFieldContainer[int]
    stream_inputs: _containers.RepeatedScalarFieldContainer[bool]
    stream_outputs: _containers.RepeatedScalarFieldContainer[bool]
    stream_weights: int
    stride: _containers.RepeatedScalarFieldContainer[int]
    stride_cols: int
    stride_depth: int
    stride_rows: int
    use_uram: bool
    weight_quant: quant_format
    weight_t: fixed_point
    def __init__(self, batch_size: _Optional[int] = ..., rows_in: _Optional[int] = ..., cols_in: _Optional[int] = ..., depth_in: _Optional[int] = ..., channels_in: _Optional[int] = ..., rows_out: _Optional[int] = ..., cols_out: _Optional[int] = ..., depth_out: _Optional[int] = ..., channels_out: _Optional[int] = ..., coarse: _Optional[int] = ..., coarse_in: _Optional[int] = ..., coarse_out: _Optional[int] = ..., mem_bw_in: _Optional[float] = ..., mem_bw_out: _Optional[float] = ..., data_t: _Optional[_Union[fixed_point, _Mapping]] = ..., input_t: _Optional[_Union[fixed_point, _Mapping]] = ..., output_t: _Optional[_Union[fixed_point, _Mapping]] = ..., input_quant: _Optional[_Union[quant_format, _Mapping]] = ..., output_quant: _Optional[_Union[quant_format, _Mapping]] = ..., stream_inputs: _Optional[_Iterable[bool]] = ..., stream_outputs: _Optional[_Iterable[bool]] = ..., use_uram: bool = ..., ports_in: _Optional[int] = ..., ports_out: _Optional[int] = ..., rows_in_array: _Optional[_Iterable[int]] = ..., cols_in_array: _Optional[_Iterable[int]] = ..., depth_in_array: _Optional[_Iterable[int]] = ..., channels_in_array: _Optional[_Iterable[int]] = ..., rows_out_array: _Optional[_Iterable[int]] = ..., cols_out_array: _Optional[_Iterable[int]] = ..., depth_out_array: _Optional[_Iterable[int]] = ..., channels_out_array: _Optional[_Iterable[int]] = ..., mem_bw_in_array: _Optional[_Iterable[float]] = ..., mem_bw_out_array: _Optional[_Iterable[float]] = ..., pad_top: _Optional[int] = ..., pad_right: _Optional[int] = ..., pad_bottom: _Optional[int] = ..., pad_left: _Optional[int] = ..., pad_front: _Optional[int] = ..., pad_back: _Optional[int] = ..., kernel_rows: _Optional[int] = ..., kernel_cols: _Optional[int] = ..., kernel_depth: _Optional[int] = ..., kernel_size: _Optional[_Iterable[int]] = ..., stride_rows: _Optional[int] = ..., stride_cols: _Optional[int] = ..., stride_depth: _Optional[int] = ..., stride: _Optional[_Iterable[int]] = ..., filters: _Optional[int] = ..., groups: _Optional[int] = ..., fine: _Optional[int] = ..., coarse_group: _Optional[int] = ..., weight_t: _Optional[_Union[fixed_point, _Mapping]] = ..., acc_t: _Optional[_Union[fixed_point, _Mapping]] = ..., bias_t: _Optional[_Union[fixed_point, _Mapping]] = ..., weight_quant: _Optional[_Union[quant_format, _Mapping]] = ..., bias_quant: _Optional[_Union[quant_format, _Mapping]] = ..., stream_weights: _Optional[int] = ..., on_chip_addr_range: _Optional[int] = ..., off_chip_buffer_size: _Optional[int] = ..., off_chip_interval: _Optional[int] = ..., sparsity: _Optional[float] = ..., has_bias: bool = ..., block_floating_point: bool = ..., split: _Optional[_Iterable[int]] = ...) -> None: ...

class partition(_message.Message):
    __slots__ = ["batch_size", "gen_last_width", "id", "input_nodes", "layers", "output_nodes", "ports", "weights_reloading_factor", "weights_reloading_layer"]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    GEN_LAST_WIDTH_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_NODES_FIELD_NUMBER: _ClassVar[int]
    LAYERS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_NODES_FIELD_NUMBER: _ClassVar[int]
    PORTS_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_RELOADING_FACTOR_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_RELOADING_LAYER_FIELD_NUMBER: _ClassVar[int]
    batch_size: int
    gen_last_width: int
    id: int
    input_nodes: _containers.RepeatedScalarFieldContainer[str]
    layers: _containers.RepeatedCompositeFieldContainer[layer]
    output_nodes: _containers.RepeatedScalarFieldContainer[str]
    ports: int
    weights_reloading_factor: int
    weights_reloading_layer: str
    def __init__(self, id: _Optional[int] = ..., ports: _Optional[int] = ..., batch_size: _Optional[int] = ..., input_nodes: _Optional[_Iterable[str]] = ..., output_nodes: _Optional[_Iterable[str]] = ..., weights_reloading_factor: _Optional[int] = ..., weights_reloading_layer: _Optional[str] = ..., gen_last_width: _Optional[int] = ..., layers: _Optional[_Iterable[_Union[layer, _Mapping]]] = ...) -> None: ...

class partitions(_message.Message):
    __slots__ = ["partition"]
    PARTITION_FIELD_NUMBER: _ClassVar[int]
    partition: _containers.RepeatedCompositeFieldContainer[partition]
    def __init__(self, partition: _Optional[_Iterable[_Union[partition, _Mapping]]] = ...) -> None: ...

class quant_format(_message.Message):
    __slots__ = ["scale", "scale_t", "shift", "shift_t"]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    SCALE_T_FIELD_NUMBER: _ClassVar[int]
    SHIFT_FIELD_NUMBER: _ClassVar[int]
    SHIFT_T_FIELD_NUMBER: _ClassVar[int]
    scale: _containers.RepeatedScalarFieldContainer[float]
    scale_t: fixed_point
    shift: _containers.RepeatedScalarFieldContainer[int]
    shift_t: fixed_point
    def __init__(self, scale_t: _Optional[_Union[fixed_point, _Mapping]] = ..., shift_t: _Optional[_Union[fixed_point, _Mapping]] = ..., scale: _Optional[_Iterable[float]] = ..., shift: _Optional[_Iterable[int]] = ...) -> None: ...

class stream(_message.Message):
    __slots__ = ["buffer_depth", "coarse", "name", "node"]
    BUFFER_DEPTH_FIELD_NUMBER: _ClassVar[int]
    COARSE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    NODE_FIELD_NUMBER: _ClassVar[int]
    buffer_depth: int
    coarse: int
    name: str
    node: str
    def __init__(self, name: _Optional[str] = ..., coarse: _Optional[int] = ..., buffer_depth: _Optional[int] = ..., node: _Optional[str] = ...) -> None: ...

class layer_type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
