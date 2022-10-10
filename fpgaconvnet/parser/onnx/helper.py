import onnx
import onnxruntime
import onnx.numpy_helper
import numpy as np

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

def update_batch_size(model, batch_size): # from https://github.com/microsoft/onnxruntime/issues/1467#issuecomment-514322927
    # change input batch size
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = batch_size
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = batch_size
    # clear value info
    model.graph.ClearField('value_info')
    # run shape inference
    return onnx.shape_inference.infer_shapes(model)

def get_model_node(model, name):
    return next(filter(lambda x: x.name == name, model.graph.node))

def get_model_value_info(model, name):
    return next(filter(lambda x: x.name == name, model.graph.value_info))

def get_model_input(model, name):
    return next(filter(lambda x: x.name == name, model.graph.input))

def get_model_output(model, name):
    return next(filter(lambda x: x.name == name, model.graph.output))

def get_model_initializer(model, name, to_tensor=True):
    init = next(filter(lambda x: x.name == name, model.graph.initializer))
    if to_tensor:
        return onnx.numpy_helper.to_array(init)
    else:
        return init

def get_input_shape(model, name):
    # get inputs and outputs
    all_tensors = [ *model.graph.input, *model.graph.output, *model.graph.value_info ]
    input = next(filter(lambda x: x.name == name, all_tensors))
    return [ x.dim_value for x in input.type.tensor_type.shape.dim ]

def format_attr(attribute):
    attr_out = {}
    for attr in attribute:
        if attr.type == 7: # (INTS) TODO: find enumeration
            attr_out[attr.name] = [ int(i) for i in attr.ints ]
        elif attr.type == 2: #(INT)
            attr_out[attr.name] = attr.i
    return attr_out

def format_onnx_name(node):
    # get baseline name
    name = node.output[0].rstrip(":0").rstrip("_Y")
    # replace all invalid characters in the layer name
    name = name.replace("/","_").replace(":","_").replace(" ","_").replace("-","_")
    if name.isnumeric(): # preprend with type to avoid macro issue
        return f"{node.op_type}{name}"
    else:
        return name

def check_model_equivalence(model_a, model_b, batch_size=32):

    # inference sessions
    model_a_ort = onnxruntime.InferenceSession(model_a.SerializeToString())
    model_b_ort = onnxruntime.InferenceSession(model_b.SerializeToString())

    # get input shapes
    model_a_input_shape = model_a_ort.get_inputs()[0].shape
    model_b_input_shape = model_b_ort.get_inputs()[0].shape

    # first check they have the same input shape
    assert model_a_input_shape == model_b_input_shape, "ERROR: input shapes are not equivalent"

    # generate random data for the input
    input_data = np.random.uniform(-1, 1,
            size=(batch_size, *model_a_input_shape[1:])).astype(np.float32)

    # execute model a
    model_a_input_name = model_a_ort.get_inputs()[0].name
    model_a_output_name = model_a_ort.get_outputs()[0].name
    model_a_output = np.array(model_a_ort.run([model_a_output_name],
        { model_a_input_name: input_data}))

    # execute model b
    model_b_input_name = model_b_ort.get_inputs()[0].name
    model_b_output_name = model_b_ort.get_outputs()[0].name
    model_b_output = np.array(model_b_ort.run([model_b_output_name],
        { model_b_input_name: input_data}))

    # check that outputs are the same shape
    assert model_a_output.shape == model_b_output.shape, "ERROR: output shapes are not equivalent"

    # check that the data is (roughly) equivalent
    assert np.allclose(model_a_output, model_b_output, atol=0.00001), "ERROR: outputs are not equal"
