import matplotlib.pyplot as plt
import numpy as np
import os
import pydot
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

def plot_latency_per_layer(self, output_path=None):

    # create figure and axis
    fig, axs = plt.subplots(len(self.partitions), 1, squeeze=False)

    # iterate over partitions
    for index, partition in enumerate(self.partitions):

        # get layer names
        nodes = [ node for node in partition.graph.nodes ]

        # get latency per layer
        latency = [ partition.graph.nodes[node]["hw"].latency() for node in nodes ]

        # plot latency
        axs[index,0].bar(np.arange(len(nodes)), latency)
        axs[index,0].set_xticks(np.arange(len(nodes)), labels=nodes, rotation=90)

        # add axis labels
        axs[index,0].set_ylabel("Latency (cycles)")
        axs[index,0].set_title(f"Partition: {index}")

        # add grid
        axs[index,0].grid(axis="y")

    if output_path == None:
        plt.show()

def plot_percentage_resource_per_layer_type(self, output_path=None):

    # define resource types
    RSC_TYPES = [ "LUT", "FF", "BRAM", "DSP" ]

    # create figure and axis
    fig, axs = plt.subplots(len(self.partitions), 4, squeeze=False)

    # iterate over partitions
    for index, partition in enumerate(self.partitions):

        # get layer names
        nodes = [ node for node in partition.graph.nodes ]
        node_types = list(set([ partition.graph.nodes[node]["type"] \
                    for node in nodes ]))

        # iterate over resource types
        for rsc_index, rsc_type in enumerate(RSC_TYPES):

            # get resources per layer
            rsc = [ partition.graph.nodes[node]["hw"].resource()[rsc_type] \
                    for node in nodes ]

            # get resources per layer_type
            rsc_per_type = [0]*len(node_types)
            for i, node_type in enumerate(node_types):
                for j, node in enumerate(nodes):
                    if partition.graph.nodes[node]["type"] == node_type:
                        rsc_per_type[i] += rsc[j]

            # plot resources
            patches, _ = axs[index,rsc_index].pie(rsc_per_type)

            # create a legend
            if rsc_type == RSC_TYPES[-1]:
                axs[index,rsc_index].legend(patches, node_types, loc="right")

            # add axis labels
            # axs[index,0].set_ylabel("Latency (cycles)")
            axs[index,rsc_index].set_title(f"{rsc_type} (p{index})")

            # add grid
            axs[index,0].grid(axis="y")

    if output_path == None:
        plt.show()

def visualise_partitions_nx(self, output_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        for file in os.listdir(output_path):
            os.remove(os.path.join(output_path, file))

    for partition_index in range(len(self.partitions)):
        graph = self.partitions[partition_index].graph
        g = pydot.Dot(graph_type='digraph')
        g.set_node_defaults(shape='record')
        for node in graph.nodes():
            if node == self.partitions[partition_index].wr_layer:
                wr_factor = self.partitions[partition_index].wr_factor
            else:
                wr_factor = None
            node_type  = graph.nodes[node]['type'].name
            rows_in       = graph.nodes[node]['hw'].rows_in()
            cols_in       = graph.nodes[node]['hw'].cols_in()
            channels_in   = graph.nodes[node]['hw'].channels_in()
            coarse_in     = graph.nodes[node]['hw'].coarse_in
            rows_out       = graph.nodes[node]['hw'].rows_out()
            cols_out       = graph.nodes[node]['hw'].cols_out()
            channels_out   = graph.nodes[node]['hw'].channels_out()
            coarse_out     = graph.nodes[node]['hw'].coarse_out
            if graph.nodes[node]['type'] in [ LAYER_TYPE.Split, LAYER_TYPE.Chop ]:
                rows_out = [rows_out] * graph.nodes[node]['hw'].ports_out
                cols_out = [cols_out] * graph.nodes[node]['hw'].ports_out
                channels_out = [channels_out] * graph.nodes[node]['hw'].ports_out
            if graph.nodes[node]['type'] in [ LAYER_TYPE.EltWise, LAYER_TYPE.Concat ]:
                rows_in = [rows_in] * graph.nodes[node]['hw'].ports_in
                cols_in = [cols_in] * graph.nodes[node]['hw'].ports_in
                channels_in = [channels_in] * graph.nodes[node]['hw'].ports_in
            if wr_factor == None:
                g.add_node(pydot.Node(node,
                    label="{{ {node}|type: {type} \n dim in: [{rows_in}, {cols_in}, {channels_in}]  \n dim out: [{rows_out}, {cols_out}, {channels_out}]  \n coarse in: {coarse_in} \| coarse out: {coarse_out}}}".format(
                    node=node,
                    type=node_type,
                    rows_in=rows_in,
                    cols_in=cols_in,
                    channels_in=channels_in,
                    coarse_in=coarse_in,
                    rows_out=rows_out,
                    cols_out=cols_out,
                    channels_out=channels_out,
                    coarse_out=coarse_out)))
            else:
                g.add_node(pydot.Node(node,
                    label="{{ {node}|type: {type} \n dim in: [{rows_in}, {cols_in}, {channels_in}]  \n dim out: [{rows_out}, {cols_out}, {channels_out}]  \n coarse in: {coarse_in} \| coarse out: {coarse_out} \n wr factor: {wr_factor}}}".format(
                    node=node,
                    type=node_type,
                    rows_in=rows_in,
                    cols_in=cols_in,
                    channels_in=channels_in,
                    coarse_in=coarse_in,
                    rows_out=rows_out,
                    cols_out=cols_out,
                    channels_out=channels_out,
                    coarse_out=coarse_out,
                    wr_factor=wr_factor)))
            for edge in graph[node]:
                g.add_edge(pydot.Edge(node,edge,splines="line"))
        g.write_png(os.path.join(output_path, f"partition_{partition_index}.png"))