import copy
import onnx
import onnxruntime
import onnx.utils
import onnx.numpy_helper
from onnx import version_converter
import onnxoptimizer as optimizer
from onnx.tools import update_model_dims
from itertools import repeat
from collections.abc import Iterable
import numpy as np

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

def load(filepath,fuse_bn=True):
    model = onnx.load(filepath)
    onnx.checker.check_model(model)
    onnx.helper.strip_doc_string(model)
    add_input_from_initializer(model) #Seems to be necessary for conv layers from pytorch (at least)
    model = onnx.shape_inference.infer_shapes(model)
    # model = onnx.utils.polish_model(model)
    passes = [
            "extract_constant_to_initializer",
            "eliminate_unused_initializer",
            "eliminate_nop_transpose",
            "eliminate_nop_pad",
            "fuse_consecutive_transposes",
            "fuse_transpose_into_gemm",
            "fuse_matmul_add_bias_into_gemm",
    ]
    if fuse_bn:
        passes.append("fuse_bn_into_conv")
    model = optimizer.optimize(model, passes=passes)
    model = convert_matmul_to_gemm(model)
    onnx.checker.check_model(model)
    return model

def update_batch_size(model, batch_size): # from https://github.com/microsoft/onnxruntime/issues/1467#issuecomment-514322927
    # change input batch size
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = batch_size
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = batch_size
    # clear value info
    model.graph.ClearField('value_info')
    # run shape inference
    return onnx.shape_inference.infer_shapes(model)

def _format_name(name):
    return name.rstrip(":0").rstrip("_Y")

def _name(node):
    #return _format_name( node.name if node.name else node.output[0] )
    return _format_name( node.output[0] )

def get_model_node(model, name):
    for node in model.graph.node:
        if _name(node) == name: # formatted match
            return node

def get_model_value_info(model, name):
    for node in model.graph.value_info:
        if _format_name(node.name) == name: # formatted match
            return node

def get_model_input(model, name):
    for node in model.graph.input:
        if node.name == name: # exact match
            return node

def get_model_initializer(model, name, to_tensor=True):
    for node in model.graph.initializer:
        if node.name == name: # exact match
            if to_tensor:
                return onnx.numpy_helper.to_array(node)
            else:
                return node

def _format_attr(attribute):
    attr_out = {}
    for attr in attribute:
        if attr.type == 7: # (INTS) TODO: find enumeration
            attr_out[attr.name] = [ int(i) for i in attr.ints ]
        elif attr.type == 2: #(INT)
            attr_out[attr.name] = attr.i
    return attr_out

def _out_dim(model, name):
    dim = [0,0,0]
    value_info = get_model_value_info(model, name)
    if len(value_info.type.tensor_type.shape.dim) == 4:
        #dim[0] = int(node.type.tensor_type.shape.dim[0].dim_value) # batch size
        dim[0] = int(value_info.type.tensor_type.shape.dim[1].dim_value) # channels
        dim[1] = int(value_info.type.tensor_type.shape.dim[2].dim_value) # rows
        dim[2] = int(value_info.type.tensor_type.shape.dim[3].dim_value) # cols
        return dim
    elif len(value_info.type.tensor_type.shape.dim) == 2:
        #dim[0] = int(node.type.tensor_type.shape.dim[0].dim_value) # batch size
        dim[0] = int(value_info.type.tensor_type.shape.dim[1].dim_value) # channels
        dim[1] = 1 # rows
        dim[2] = 1 # cols
        return dim

def gen_layer_name(graph, layer_name): # layer in protobuf form
    # looks through graph to find node, to get type
    layer_type = graph.nodes[layer_name]['type']
    #FIXME bit of a hacky way to get a good type name
    layer_type_str = str(layer_type)[11:].upper() # remove 'LAYER_TYPE.'
    # replace all invalid characters in the layer name
    layer_name = layer_name.replace("/","_").replace(":","_")
    if layer_name.isnumeric(): # preprend with type to avoid macro issue
        return f'{layer_type_str}{layer_name}'
    else:
        return layer_name

def convert_matmul_to_gemm(model):
    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):
        if node.op_type == "MatMul":
            # update the weights
            init = get_model_initializer(model, node.input[1], to_tensor=False)
            init_index = list(model.graph.initializer).index(init)
            weights = onnx.numpy_helper.to_array(init)
            weights = np.swapaxes(weights,0,1)
            new_init = onnx.helper.make_tensor(
                name=node.input[1],
                data_type=init.data_type,
                dims=weights.shape,
                vals=weights.flatten().tolist())
            # update weight's value info
            init_value_info = get_model_input(model, node.input[1])
            init_value_info_index = list(model.graph.input).index(init_value_info)
            new_init_value_info = onnx.helper.make_tensor_value_info(
                    node.input[1],
                    onnx.TensorProto.FLOAT,
                    weights.shape)
            # update the graph
            model.graph.initializer.remove(init)
            model.graph.initializer.insert(init_index,new_init)
            model.graph.input.remove(init_value_info)
            model.graph.input.insert(init_value_info_index, new_init_value_info)
            # add an empty bias term
            new_bias = onnx.helper.make_tensor(
                name="_".join([node.input[1],"bias"]),
                data_type=init.data_type,
                dims=(weights.shape[0],),
                vals=np.zeros(weights.shape[0]).flatten().tolist())
            new_bias_value_info = onnx.helper.make_tensor_value_info(
                    new_bias.name,
                    onnx.TensorProto.FLOAT,
                    [weights.shape[0]])
            # update the graph
            model.graph.initializer.insert(-1,new_bias)
            model.graph.input.insert(-1,new_bias_value_info)
            # create a new matmul node
            new_node = onnx.helper.make_node(
                "Gemm",
                name=node.name,
                inputs=[*node.input, "_".join([node.input[1],"bias"])],
                outputs=node.output,
                alpha=1.0,
                beta=1.0,
                transA=0,
                transB=1
            )
            # remove old node and add new one
            model.graph.node.remove(node)
            model.graph.node.insert(index, new_node)
    # return the new model
    return model




