from dataclasses import dataclass
import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpbinary import FpBinary

@dataclass
class FixedPoint:
    width: int
    binary_point: int

    def to_protobuf(self, fixed_point):
        fixed_point.width = self.width
        fixed_point.binary_point = self.binary_point

    def to_dict(self):
        return {
            "width": self.width,
            "binary_point": self.binary_point
        }

    def apply(self, val):
        return FpBinary(int_bits=self.width-self.binary_point,
            frac_bits=self.binary_point, signed=True, value=val)
