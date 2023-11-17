import toml

from enum import Enum
from dataclasses import dataclass

class BACKEND(Enum):
    """
    An enumeration of the supported backends for the FPGA ConvNet model.

    Attributes:
        HLS (int): The High-Level Synthesis (HLS) backend.
        CHISEL (int): The Chisel backend.
    """
    HLS = 0
    CHISEL = 1

class DIMENSIONALITY(Enum):

    """
    An enumeration of the supported dimensionalities for the FPGA ConvNet model.

    Attributes:
        TWO (int): The 2D backend.
        THREE (int): The 3D backend.
    """
    TWO = 2
    THREE = 3

class SPARSITY(Enum):

    """
    An enumeration of the supported sparsity options for the FPGA ConvNet model.

    Attributes:
        SPARSE (int): The sparse backend.
        DENSE (int): The dense backend.
    """
    SPARSE = 0
    DENSE = 1

class DATA_PACKING(Enum):

    """
    An enumeration of the supported data packing options for the FPGA ConvNet model.

    Attributes:
        PACKED (int): The packed backend.
        UNPACKED (int): The unpacked backend.
    """
    PACKED = 0
    UNPACKED = 1

class LATENCY(Enum):

    """
    An enumeration of the supported latency options for the FPGA ConvNet model.

    Attributes:
        LATENCY (int): The latency backend.
        THROUGHPUT (int): The throughput backend.
    """
    LATENCY = 0
    THROUGHPUT = 1

class DATA_TYPE(Enum):

    """
    An enumeration of the supported data types for the FPGA ConvNet model.

    Attributes:
        FLOAT (int): The float backend.
        FIXED (int): The fixed backend.
    """
    FLOAT = 0
    FIXED = 1
    QDQ = 2
    BLOCK_FLOAT = 3

class WEIGHTS(Enum):
    """
    An enumeration of the different weight storage options for a neural network.

    Attributes:
        ON_CHIP (int): Indicates that the weights should be stored on the FPGA chip.
        STREAM (int): Indicates that the weights should be streamed from the host to the FPGA.
    """
    ON_CHIP = 0
    STREAM = 1

class RESOURCE_MODELLING(Enum):
    """
    Enum for different resource modelling techniques.

    Attributes:
        LINEAR_REGRESSION (int): Use linear regression to model the resource usage.
        XGBOOST (int): Use XGBoost to model the resource usage.
    """
    LINEAR_REGRESSION = 0
    XGBOOST = 1

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
    sparsity: SPARSITY = SPARSITY.DENSE
    data_packing: DATA_PACKING = DATA_PACKING.UNPACKED
    latency: LATENCY = LATENCY.THROUGHPUT
    data_type: DATA_TYPE = DATA_TYPE.FIXED
    weights: WEIGHTS = WEIGHTS.ON_CHIP
    resource_modelling: RESOURCE_MODELLING = RESOURCE_MODELLING.LINEAR_REGRESSION

    def __str__(self):
        """
        Returns a string representation of the architecture flags.

        Returns:
            str: A string representation of the architecture flags.
        """
        return str(self.backend)+str(self.dimensionality)+str(self.sparsity)+str(self.data_packing)+str(self.latency)+str(self.data_type)+str(self.weights)

    def load_from_toml(self, filepath):
        """
        Loads the flags from a TOML file. An example toml file is shown below:

        ```
        [flags]
        backend = "hls"
        dimensionality = "two"
        sparsity = "dense"
        data_packing = "unpacked"
        latency = "throughput"
        data_type = "fixed"
        weights = "on-chip"
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
        self.sparsity           = SPARSITY[toml_dict['sparsity'].upper()]
        self.data_packing       = DATA_PACKING[toml_dict['data_packing'].upper()]
        self.latency            = LATENCY[toml_dict['latency'].upper()]
        self.data_type          = DATA_TYPE[toml_dict['data_type'].upper()]
        self.weights            = WEIGHTS[toml_dict['weights'].upper()]
        self.resource_modelling = RESOURCE_MODELLING[toml_dict['resource_modelling'].upper()]


