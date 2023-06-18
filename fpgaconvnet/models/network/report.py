import json
import datetime
import numpy as np

from dataclasses import asdict
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

def create_report(self, output_path):
    # create report dictionary
    total_operations = sum([partition.get_total_operations() for partition in self.partitions])
    #total_dsps = np.average([partition.get_resource_usage()["DSP"]) for partition in self.partitions])
    report = {}
    report = {
        "name" : self.name,
        "date_created" : str(datetime.datetime.now()),
        "total_iterations" : 0, # TODO
        "platform" : asdict(self.platform),
        "total_operations" : float(total_operations),
        "network" : {
            "memory_usage" : self.get_memory_usage_estimate(),
            "multi_fpga" : self.multi_fpga,
            "performance" : {
                "latency" : self.get_latency(),
                "throughput" : self.get_throughput(),
                "performance" : total_operations/self.get_latency(),
                "cycles" : self.get_cycle(),
                "partition_delay" : self.get_partition_delay()
            },
            "num_partitions" : len(self.partitions),
            "max_resource_usage" : {
                "LUT" : max([ partition.get_resource_usage()["LUT"] for partition in self.partitions ]),
                "FF" : max([ partition.get_resource_usage()["FF"] for partition in self.partitions ]),
                "BRAM" : max([ partition.get_resource_usage()["BRAM"] for partition in self.partitions ]),
                "DSP" : max([ partition.get_resource_usage()["DSP"] for partition in self.partitions ])
            },
            "sum_resource_usage" : {
                "LUT" : int(np.sum([ partition.get_resource_usage()["LUT"] for partition in self.partitions ])),
                "FF" : int(np.sum([ partition.get_resource_usage()["FF"] for partition in self.partitions ])),
                "BRAM" : int(np.sum([ partition.get_resource_usage()["BRAM"] for partition in self.partitions ])),
                "DSP" : int(np.sum([ partition.get_resource_usage()["DSP"] for partition in self.partitions ]))
            }
        }
    }

    if self.platform.get_uram() > 0:
        report["network"]["max_resource_usage"]["URAM"] = max([ partition.get_resource_usage()["URAM"] for partition in self.partitions ])
        report["network"]["sum_resource_usage"]["URAM"] = int(np.sum([ partition.get_resource_usage()["URAM"] for partition in self.partitions ]))
    
    # add information for each partition
    report["partitions"] = {}
    for i in range(len(self.partitions)):
        # get some information on the partition
        resource_usage = self.partitions[i].get_resource_usage()
        latency = self.partitions[i].get_latency(self.platform.board_freq)
        # add partition information
        report["partitions"][i] = {
            "partition_index" : i,
            "batch_size" : self.partitions[i].batch_size,
            "num_layers" : len(self.partitions[i].graph.nodes()),
            "latency" : latency,
            "cycles" : self.partitions[i].get_cycle(),
            "weights_reloading_factor" : self.partitions[i].wr_factor,
            "weights_reloading_layer" : self.partitions[i].wr_layer,
            "resource_usage" : {
                "LUT" : resource_usage["LUT"],
                "FF" : resource_usage["FF"],
                "BRAM" : resource_usage["BRAM"],
                "DSP" : resource_usage["DSP"]
            },
            "bandwidth" : {
                "in (GB/s)" : self.partitions[i].get_bandwidth_in(self.platform.board_freq), 
                "out (GB/s)" : self.partitions[i].get_bandwidth_out(self.platform.board_freq), 
                "weight (bits per cycle)": self.partitions[i].get_bandwidth_weight()
            }
        }
        if self.platform.get_uram() > 0:
            report["partitions"][i]["resource_usage"]["URAM"] = resource_usage["URAM"]

        # add information for each layer of the partition
        report["partitions"][i]["layers"] = {}
        for node in self.partitions[i].graph.nodes():
            hw = self.partitions[i].graph.nodes[node]['hw']
            resource_usage = hw.resource()
            report["partitions"][i]["layers"][node] = {
                "type" : str(self.partitions[i].graph.nodes[node]['type']),
                "interval" : hw.latency(), #TODO
                "cycle" : hw.latency(),
                "resource_usage" : {
                    "LUT" : resource_usage["LUT"],
                    "FF" : resource_usage["FF"],
                    "BRAM" : resource_usage["BRAM"],
                    "DSP" : resource_usage["DSP"]
                }
            }
            if "URAM" in resource_usage.keys():
                report["partitions"][i]["layers"][node]["resource_usage"]["URAM"] = resource_usage["URAM"]
            if self.partitions[i].graph.nodes[node]["type"] == LAYER_TYPE.Convolution:
                report["partitions"][i]["layers"][node]["resource_usage"]["stream_bw"] = hw.stream_bw()
    # save as json
    with open(output_path,"w") as f:
        json.dump(report,f,indent=2)


