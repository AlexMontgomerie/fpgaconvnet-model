import os
import toml
from typing import ClassVar, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class PlatformBase:

    part: str
    resources: dict[str,int]
    reconf_time: float
    board_freq: float = 100.0
    mem_bw: float = 10.0
    port_width: int = 512

    family: ClassVar[str]
    resource_types: ClassVar[list[str]]

    def __post_init__(self):

        # check resource types
        assert all([rsc in self.resource_types for rsc in self.resources.keys()]), f"resource types must be one of {self.resource_types}"

        # check resource values
        assert all([rsc_val >= 0 for rsc_val in self.resources.values()]), "resource values must be positive"

        # check reconfiguration time
        assert self.reconf_time >= 0, "reconfiguration time must be positive"

        # check board frequency
        assert self.board_freq > 0, "board frequency must be positive"

        # check memory bandwidth
        assert self.mem_bw > 0, "memory bandwidth must be positive"

        # check port width
        assert self.port_width > 0, "port width must be positive"

    @classmethod
    def from_toml(cls, platform_path: str):

        # make sure toml configuration
        assert os.path.splitext(platform_path)[1] == ".toml", "must be a TOML configuration file"

        # parse platform configuration toml file
        with open(platform_path, "r") as f:
            conf = toml.load(f)

        # initialise platform
        return cls(
            part=conf["device"]["part"],
            resources=conf["resources"],
            reconf_time=conf["system"]["reconfiguration_time"],
            board_freq=conf["system"]["board_frequency"],
            mem_bw=conf["system"]["memory_bandwidth"],
            port_width=conf["system"]["port_width"]
        )

    def get_resource(self, rsc_type: str) -> int:
        if rsc_type not in self.resource_types:
            raise ValueError(f"resource type {rsc_type} not supported by family {self.family} (should be one of {self.resource_types})")
        return self.resources[rsc_type]

    def get_resource_database_filters(self, scale: float = 0.8) -> dict[str,Union[str,dict[str,int]]]:
        filters = { "fpga": self.part }
        for rsc_type, rsc_max in self.resources.items():
            filters[f"resource.{rsc_type}"] = { "$exists": True, "$lt": int(scale*self.get_resource(rsc_type)) }
        return filters

    def update_from_toml(self, platform_path: str):

        # make sure toml configuration
        assert os.path.splitext(platform_path)[1] == ".toml", "must be a TOML configuration file"

        # parse platform configuration toml file
        with open(platform_path, "r") as f:
            conf = toml.load(f)

        # update fields
        self.part = conf["device"]["part"]

        ## resources
        for resource, val in conf["resources"].items():
            self.resources[resource] = val

        ## system
        self.board_freq  = conf["system"].get("board_frequency", 100.0) # in MHz
        self.mem_bw      = conf["system"].get("memory_bandwidth", 5.0) # in Gbps
        self.reconf_time = conf["system"].get("reconfiguration_time", 0.0) # in seconds
        self.port_width  = conf["system"].get("port_width", 512) # in bits

        ## ethernet
        if "ethernet" in conf.keys():
            self.eth_bw = conf["ethernet"]["bandwidth"] # in Gbps
            self.eth_port_width = self.calculate_eth_port_width() # in bits
            self.eth_delay = conf["ethernet"]["latency"] # in seconds

        # perform post initialisation again
        self.__post_init__()

    def calculate_eth_port_width(self):
        # equivalent ethernet port width from async fifo
        mac_head = 10  # Bytes
        ip_head = 20  # Bytes
        udp_head = 8  # Bytes
        max_packet_size = 576  # Bytes, reassembly buffer size

        # todo: apply the constraint as bandwidth instead of port width
        eff_bw = self.eth_bw * (max_packet_size - mac_head - ip_head - udp_head) / max_packet_size
        eth_port_width = eff_bw / (self.board_freq / 1000)  # in bits
        return int(eth_port_width)
