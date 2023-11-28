import toml

from enum import Enum, IntEnum, auto
from dataclasses import dataclass

class BACKEND(Enum):
    """
    An enumeration of the supported backends for the FPGA ConvNet model.

    Attributes:
        HLS (int): The High-Level Synthesis (HLS) backend.
        CHISEL (int): The Chisel backend.
    """
    HLS = auto()
    CHISEL = auto()

class DIMENSIONALITY(IntEnum):

    """
    An enumeration of the supported dimensionalities for the FPGA ConvNet model.

    Attributes:
        TWO (int): The 2D backend.
        THREE (int): The 3D backend.
    """
    TWO = 2
    THREE = 3

@dataclass
class Architecture:
    """
    A dataclass that holds all of the flags for the FPGA ConvNet model.

    Attributes:
        backend (BACKEND): The backend to use for the FPGA ConvNet model.
        dimensionality (DIMENSIONALITY): The dimensionality to use for the FPGA ConvNet model.
        sparsity (SPARSITY): The sparsity option to use for the FPGA ConvNet model.
        data_packing (DATA_PACKING): The data packing option to use for the FPGA ConvNet model.
        latency (LATENCY): The latency option to use for the FPGA ConvNet model.
        data_type (DATA_TYPE): The data type to use for the FPGA ConvNet model.
        weights (WEIGHTS): The weight storage option to use for the FPGA ConvNet model.
    """
    backend: BACKEND = BACKEND.HLS
    dimensionality: DIMENSIONALITY = DIMENSIONALITY.TWO

    def __str__(self):
        """
        Returns a string representation of the architecture flags.

        Returns:
            str: A string representation of the architecture flags.
        """
        return f"backend.{self.backend.value}.dimensionality.{self.dimensionality}"

    def load_from_toml(self, filepath):
        """
        Loads the flags from a TOML file. An example toml file is shown below:

        ```
        [flags]
        backend = "hls"
        dimensionality = "two"
        ```

        Args:
            filepath (str): The path to the TOML file.
        """

        # open the TOML file
        with open(filepath, 'r') as f:
            # load the TOML file into a dictionary
            toml_dict = toml.load(f)["flags"]

        # set all the class attributes to the values in the dictionary
        self.backend            = BACKEND[toml_dict['backend'].upper()]
        self.dimensionality     = DIMENSIONALITY[toml_dict['dimensionality'].upper()]

