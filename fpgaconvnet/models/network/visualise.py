import matplotlib.pyplot as plt
import numpy as np

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


