import os
import toml
from dataclasses import dataclass, field

@dataclass
class Platform:
    # device specification
    part: str = ""
    board: str = ""
    # resources
    resources: dict = field(default_factory=dict)
    # system
    board_freq: float = 100.0
    mem_bw: float = 10.0
    mem_bw_wpc: float = 6.25
    reconf_time: float = 10.0
    axi_ports: int = 2
    axi_data_width: int = 128

    def __post_init__(self):
        self.get_name()
        self.get_family()
        self.get_mem_bw_wpc()

    def get_family(self):
        pass

    def get_name(self):
        if self.board:
            self.name = self.board.split(":")[1]
            return self.name

    def get_mem_bw_wpc(self):
        # memory bandwidth expressed in words per cycle (assuming 16-bit words)
        self.mem_bw_wpc = (self.mem_bw * 1e9) / (self.board_freq * 1e6 * 16)

    def get_dsp(self):
        return self.resources.get("DSP", 0)

    def get_bram(self):
        return self.resources.get("BRAM", 0)

    def get_lut(self):
        return self.resources.get("LUT", 0)

    def get_ff(self):
        return self.resources.get("FF", 0)

    def update(self, platform_path):

        # make sure toml configuration
        assert os.path.splitext(platform_path)[1] == ".toml", "must be a TOML configuration file"

        # parse platform configuration toml file
        with open(platform_path, "r") as f:
            conf = toml.load(f)

        # update fields
        ## device
        self.part = conf["device"]["part"]
        self.board = conf["device"]["board"]

        ## resources
        for resource, val in conf["resources"].items():
            self.resources[resource] = val

        ## system
        self.board_freq = conf["system"].get("board_frequency", 100.0)
        self.mem_bw = conf["system"].get("memory_bandwidth", 5.0)
        self.reconf_time = conf["system"].get("reconfiguration_time", 0.0)
        self.axi_ports = conf["system"].get("axi_ports", 2)
        self.axi_data_width = conf["system"].get("axi_data_width", 128)

        # perform post initialisation again
        self.__post_init__()

    def get_memory_bandwidth(self):
        pl_bw = self.axi_ports*self.axi_data_width*self.board_freq/1000.0
        return min(pl_bw, self.mem_bw)
