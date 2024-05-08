from google.protobuf.json_format import MessageToDict

import fpgaconvnet.tools.layer_enum

from fpgaconvnet.tools.layer_enum import LAYER_TYPE
import fpgaconvnet.tools.graphs as graphs
import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2

# from fpgaconvnet.models.layers import Layer, Layer3D, MultiPortLayer, MultiPortLayer3D
# from fpgaconvnet.models.layers import ConvolutionSparseLayer, ConvolutionPointwiseSparseLayer

def partition_info(self, partition, id: int = 0):

    # add partition info
    partition.id = id
    partition.ports = 1 # TODO
    partition.input_nodes.extend(self.input_nodes)
    partition.output_nodes.extend(self.output_nodes)

    partition.batch_size  = self.batch_size
    partition.weights_reloading_factor = self.wr_factor
    partition.weights_reloading_layer  = str(self.wr_layer)

    partition.gen_last_width = 16 # TODO: workout best width

    # add all layers (in order)
    for node in graphs.ordered_node_list(self.graph):

        # create layer
        layer = partition.layers.add()
        # layer.name = onnx_helper.format_onnx_name(node)
        layer.name = node

        # todo: implement these activations
        layer.type = fpgaconvnet.tools.layer_enum.to_proto_layer_type(
                self.graph.nodes[node]['type'])

        if self.graph.nodes[node]['type'] == LAYER_TYPE.EltWise:
            layer.op_type = self.graph.nodes[node]['hw'].op_type

        elif self.graph.nodes[node]['type'] in [ LAYER_TYPE.ReLU, LAYER_TYPE.Sigmoid, LAYER_TYPE.HardSigmoid, LAYER_TYPE.HardSwish]:
            layer.op_type = self.graph.nodes[node]['type'].name

        elif self.graph.nodes[node]['type'] in [LAYER_TYPE.Pooling, LAYER_TYPE.GlobalPooling]:
            layer.op_type = self.graph.nodes[node]['hw'].pool_type

        elif self.graph.nodes[node]['type'] == LAYER_TYPE.Convolution:
            hw = self.graph.nodes[node]['hw']
            layer.op_type = 'dense'
            # if type(hw) == ConvolutionPointwiseSparseLayer: layer.op_type = 'pointwise_sparse'
            # elif type(hw) == ConvolutionSparseLayer: layer.op_type = 'sparse'
            # else: layer.op_type = 'dense'

        layer.onnx_node = self.graph.nodes[node]['onnx_node']

        # nodes into layer
        prev_nodes = graphs.get_prev_nodes(self.graph, node)
        # prev_nodes = list(self.get_prev_nodes_ordered(node, i))

        if not prev_nodes:
            stream_in = layer.streams_in.add()
            stream_in.name  = "in"
            stream_in.coarse = self.graph.nodes[node]['hw'].streams_in()
            stream_in.node = node
            stream_in.buffer_depth = self.graph.nodes[node]['hw'].get_buffer_depth()
        else :
            for j, prev_node in enumerate(prev_nodes):
                stream_in = layer.streams_in.add()
                stream_in.name  = "_".join([prev_node, layer.name])
                stream_in.coarse = self.graph.nodes[node]['hw'].streams_in()
                stream_in.node = prev_node
                stream_in.buffer_depth = self.graph.nodes[node]['hw'].get_buffer_depth(j)

        # nodes out of layer
        next_nodes = graphs.get_next_nodes(self.graph, node)

        if not next_nodes:
            stream_out = layer.streams_out.add()
            stream_out.name  = "out"
            stream_out.coarse = self.graph.nodes[node]['hw'].streams_out()
            stream_out.node = node
        else:
            for j, next_node in enumerate(next_nodes):
                stream_out = layer.streams_out.add()
                stream_out.name = "_".join([layer.name, next_node])
                stream_out.coarse = self.graph.nodes[node]['hw'].streams_out()
                stream_out.node = next_node

        # add parameters
        self.graph.nodes[node]['hw'].layer_info(
                layer.parameters, batch_size=self.batch_size)

        # add weights key
        if self.graph.nodes[node]['type'] in \
                [ LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct ]:
            layer.weights_path = self.graph.nodes[node]['inputs']['weights']
            layer.bias_path    = self.graph.nodes[node]['inputs']['bias']

    # return the partition protobuf object
    return partition

def partition_info_dict(self):
    # get parameters
    partition = fpgaconvnet_pb2.partition()
    self.partition_info(partition)
    # convert to dictionary
    return MessageToDict(partition, preserving_proto_field_name=True)


