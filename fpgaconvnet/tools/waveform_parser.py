from dataclasses import dataclass
from vcdvcd import VCDVCD

@dataclass
class VCDWaveformParser:
    vcd_path: str

    def __post_init__(self):
        self.vcd = VCDVCD(self.vcd_path)
        self.signals = self.vcd.signals
        self.reference_signals = self.vcd.references_to_ids
        self.data = self.vcd.data

    def get_signal_first_edge(self, signal):
        time, value = signal.tv[0]
        if time == 0 and value == '0':
            time, value = signal.tv[1]
        return time // 2

    def get_signal_last_edge(self, signal):
        time, value = signal.tv[-1]
        return time // 2

    def get_signals_per_layer(self, layer_name):
        if "GlobalMaxPool" in layer_name:
            layer_name = "GlobalAveragePoolingFixed"
        elif "MaxPool" in layer_name:
            layer_name = "PoolingBlockFixed"
        elif "Convolution" in layer_name:
            layer_name = "ConvolutionBlockFixed"

        in_valid_signals = [signal for signal in self.signals if f"{layer_name}.io_in_0_0_valid" in signal]
        out_valid_signals = [signal for signal in self.signals if f"{layer_name}.io_out_0_0_valid" in signal]

        layer_hw_stats = {
            "in_valid_signal": in_valid_signals[0],
            "out_valid_signal": out_valid_signals[0],
        }
        return layer_hw_stats

    def get_layer_stats(self, layer_type):

        layer_hw_stats = self.get_signals_per_layer(layer_type)

        layer_hw_stats["first_in_valid_cycles"] = self.get_signal_first_edge(self.data[self.reference_signals[layer_hw_stats["in_valid_signal"]]])
        layer_hw_stats["last_in_valid_cycles"] = self.get_signal_last_edge(self.data[self.reference_signals[layer_hw_stats["in_valid_signal"]]])
        layer_hw_stats["first_out_valid_cycles"] = self.get_signal_first_edge(self.data[self.reference_signals[layer_hw_stats["out_valid_signal"]]])
        layer_hw_stats["last_out_valid_cycles"] = self.get_signal_last_edge(self.data[self.reference_signals[layer_hw_stats["out_valid_signal"]]])
        layer_hw_stats["layer_total_cycles"] = layer_hw_stats["last_out_valid_cycles"] - layer_hw_stats["first_in_valid_cycles"]
        layer_hw_stats["layer_pipeline_depth_cycles"] = layer_hw_stats["first_out_valid_cycles"] - layer_hw_stats["first_in_valid_cycles"]

        return layer_hw_stats

    def get_signals_per_module(self, module_name):
        if "Pad" in module_name:
            module_name = "PadFixed"
        elif "SlidingWindow" in module_name:
            module_name = "SlidingWindowBlockFixed"
        elif "Squeeze" in module_name:
            module_name = "SqueezeBlockFixed"
        elif "Fork" in module_name:
            module_name = "ForkBlockFixed"
        elif "Accum" in module_name:
            module_name = "AccumBlockFixed"
        elif "Glue" in module_name:
            module_name = "GlueBlockFixed"
        elif "Bias" in module_name:
            module_name = "BiasBlockFixed"
        elif "VectorDot" in module_name:
            module_name = "VectorDotBlockFixed"

        in_valid_signals = [signal for signal in self.signals if f"{module_name}.io_in_valid" in signal]
        out_valid_signals = [signal for signal in self.signals if f"{module_name}.io_out_valid" in signal]

        module_hw_stats = {
            "in_valid_signal": in_valid_signals[0],
            "out_valid_signal": out_valid_signals[0],
        }
        return module_hw_stats

    def get_module_stats(self, module_type):

        module_hw_stats = self.get_signals_per_module(module_type)

        module_hw_stats["first_in_valid_cycles"] = self.get_signal_first_edge(self.data[self.reference_signals[module_hw_stats["in_valid_signal"]]])
        module_hw_stats["last_in_valid_cycles"] = self.get_signal_last_edge(self.data[self.reference_signals[module_hw_stats["in_valid_signal"]]])
        module_hw_stats["first_out_valid_cycles"] = self.get_signal_first_edge(self.data[self.reference_signals[module_hw_stats["out_valid_signal"]]])
        module_hw_stats["last_out_valid_cycles"] = self.get_signal_last_edge(self.data[self.reference_signals[module_hw_stats["out_valid_signal"]]])
        module_hw_stats["module_total_cycles"] = module_hw_stats["last_out_valid_cycles"] - module_hw_stats["first_in_valid_cycles"]
        module_hw_stats["module_pipeline_depth_cycles"] = module_hw_stats["first_out_valid_cycles"] - module_hw_stats["first_in_valid_cycles"]

        return module_hw_stats