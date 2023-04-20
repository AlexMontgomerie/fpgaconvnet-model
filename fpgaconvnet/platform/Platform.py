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
    port_width: int = 512

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

    def get_uram(self):
        return self.resources.get("URAM", 0)

    def get_lut(self):
        return self.resources.get("LUT", 0)

    def get_ff(self):
        return self.resources.get("FF", 0)

    def get_mem_bw(self):
        return self.mem_bw

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

    def get_memory_bandwidth(self):
        pl_bw = 2*16*self.port_width*self.board_freq/1000.0 # TODO: get the data type
        return min(pl_bw, self.mem_bw)

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