import random
import copy

import onnx
import onnxruntime
import onnx.numpy_helper
import numpy as np

from fpgaconvnet.parser.onnx.onnx_model_utils import make_dim_param_fixed, make_input_shape_fixed, fix_output_shapes

onnxruntime.set_default_logger_severity(3)

def add_value_info_for_constants(model : onnx.ModelProto):
    """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph : onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.value_info}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.value_info.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)


    return add_const_value_infos_to_graph(model.graph)

def add_input_from_initializer(model : onnx.ModelProto):
    """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph : onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.input}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.input.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)


    return add_const_value_infos_to_graph(model.graph)

def update_batch_size(model, batch_size):
    # get the input shape
    input_shape = get_input_shape(model, model.graph.input[0].name)
    batch_size_param = model.graph.input[0].type.tensor_type.shape.dim[0].dim_param
    if model.graph.input[0].type.tensor_type.shape.dim[0] == "":
        print(f"CRITICAL WARNING: batch size dimension already fixed to {input_shape[0]}")
        return model
    make_dim_param_fixed(model.graph, batch_size_param, batch_size)
    return model

def get_model_node(model, name):
    try:
        return next(filter(lambda x: x.name == name, model.graph.node))
    except StopIteration:
        raise StopIteration(f"{name} does not exist in model")

def get_model_value_info(model, name):
    try:
        return next(filter(lambda x: x.name == name, model.graph.value_info))
    except StopIteration:
        raise StopIteration(f"value info for {name} does not exist in model")

def get_model_input(model, name):
    try:
        return next(filter(lambda x: x.name == name, model.graph.input))
    except StopIteration:
        raise StopIteration(f"input for {name} does not exist in model")

def get_model_output(model, name):
    try:
        return next(filter(lambda x: x.name == name, model.graph.output))
    except StopIteration:
        raise StopIteration(f"output for {name} does not exist in model")

def get_model_initializer(model, name, to_tensor=True):
    try:
        init = next(filter(lambda x: x.name == name, model.graph.initializer))
        if to_tensor:
            return onnx.numpy_helper.to_array(init)
        else:
            return init
    except StopIteration:
        return None

def get_input_shape(model, name):
    # get inputs and outputs
    all_tensors = [ *model.graph.input, *model.graph.output, *model.graph.value_info ]
    input = next(filter(lambda x: x.name == name, all_tensors))
    return [ x.dim_value for x in input.type.tensor_type.shape.dim ]

def format_attr(attribute):
    """
    implements: https://github.com/onnx/onnx/blob/72c2578c3d6e14fcf0e87db1c8379304c7f49561/onnx/onnx.proto#L122-L139
    """
    attr_out = {}
    for attr in attribute:
        match attr.type:
            case 7:
                attr_out[attr.name] = [ int(i) for i in attr.ints ]
            case 6:
                attr_out[attr.name] = [ float(i) for i in attr.floats ]
            case 3:
                attr_out[attr.name] = attr.s
            case 2:
                attr_out[attr.name] = attr.i
            case 1:
                attr_out[attr.name] = attr.f
            case _:
                assert False, f"Unsupported attribute type {attr.type}"
    return attr_out

def format_onnx_name(node):
    # get baseline name
    # name = node.output[0].rstrip(":0").rstrip("_Y")
    name = node.name #.rstrip(":0").rstrip("_Y")
    # replace all invalid characters in the layer name
    invalid_char = "/: -;"
    for c in invalid_char:
        name = name.replace(c,"_")
    # name = name.replace("/","_").replace(":","_").replace(" ","_").replace("-","_").replace(";","_")
    if name.isnumeric(): # preprend with type to avoid macro issue
        return f"{node.op_type}{name}"
    else:
        return name

def check_model_equivalence(model_a, model_b, batch_size=32, seed=2342315703):


    # type lookup
    ort_type_to_np = {
        "tensor(int8)" : np.int8,
        "tensor(float)" : np.float32
    }

    # set seeds
    np.random.seed(seed)
    random.seed(seed)

    # # add intermediate layers to outputs
    # for node in model_a.graph.node:
    #     layer_info = onnx.helper.ValueInfoProto()
    #     layer_info.name = node.output[0]
    #     model_a.graph.output.append(layer_info)
    # for node in model_b.graph.node:
    #     layer_info = onnx.helper.ValueInfoProto()
    #     layer_info.name = node.output[0]
    #     model_b.graph.output.append(layer_info)

    # inference sessions
    model_a_ort = onnxruntime.InferenceSession(model_a.SerializeToString())
    model_b_ort = onnxruntime.InferenceSession(model_b.SerializeToString())

    # get input shapes
    model_a_input_shape = model_a_ort.get_inputs()[0].shape
    model_b_input_shape = model_b_ort.get_inputs()[0].shape

    # first check they have the same input shape
    assert model_a_input_shape == model_b_input_shape, "ERROR: input shapes are not equivalent"

    # generate random data for the input
    input_data = copy.deepcopy(np.random.uniform(-1, 1,
        size=model_a_input_shape).astype(np.float32))
    # input_data = copy.deepcopy(np.random.randint(-128, 127,
    #     size=model_a_input_shape).astype(np.float32))

    # execute model a
    model_a_input_type = model_a_ort.get_inputs()[0].type
    model_a_input_name = model_a_ort.get_inputs()[0].name
    model_a_output_name = model_a_ort.get_outputs()[0].name
    model_a_output = np.array(model_a_ort.run([model_a_output_name],
        { model_a_input_name: input_data.astype(ort_type_to_np[model_a_input_type]) }))

    # execute model b
    model_b_input_type = model_b_ort.get_inputs()[0].type
    model_b_input_name = model_b_ort.get_inputs()[0].name
    model_b_output_name = model_b_ort.get_outputs()[0].name
    model_b_output = np.array(model_b_ort.run([model_b_output_name],
        { model_b_input_name: input_data.astype(ort_type_to_np[model_b_input_type]) }))

    # check that outputs are the same shape
    assert model_a_output.shape == model_b_output.shape, "ERROR: output shapes are not equivalent"

    # check that the data is (roughly) equivalent
    assert np.allclose(model_a_output, model_b_output, atol=0.00001), \
            f"ERROR: outputs are not equal ({model_a_output} != {model_b_output})"

# https://github.com/Xilinx/finn-base/blob/beae9785f2e29ef021541a3499832f3754e1026d/src/finn/core/modelwrapper.py#L370
def find_consumers(model, tensor_name):
    consumers = []
    for node in model.graph.node:
        for input_index, input_tensor in enumerate(node.input):
            if input_tensor == tensor_name:
                consumers.append((node, input_index))
    return consumers
