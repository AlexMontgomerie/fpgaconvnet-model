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
        #TODO: This is a hack to get the last edge of the signal when the valid signal is not low at the end of the simulation
        if len(signal.tv) > 2:
            time, value = signal.tv[-1]
        else:
            time = signal.endtime
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

        if layer_type == "Convolution":
            self.get_conv_modules_stats(layer_hw_stats)

        return layer_hw_stats

    def get_conv_modules_stats(self, layer_hw_stats):
        pad_in_valid_signals = [signal for signal in self.signals if f"ConvolutionBlockFixed.pad_io_in_valid" in signal][0]
        pad_out_valid_signals = [signal for signal in self.signals if f"ConvolutionBlockFixed.pad_io_out_valid" in signal][0]
        pad_stats = {
            "first_in_valid_cycles": self.get_signal_first_edge(self.data[self.reference_signals[pad_in_valid_signals]]),
            "last_in_valid_cycles": self.get_signal_last_edge(self.data[self.reference_signals[pad_in_valid_signals]]),
            "first_out_valid_cycles": self.get_signal_first_edge(self.data[self.reference_signals[pad_out_valid_signals]]),
            "last_out_valid_cycles": self.get_signal_last_edge(self.data[self.reference_signals[pad_out_valid_signals]])
        }
        pad_stats["module_total_cycles"] = pad_stats["last_out_valid_cycles"] - pad_stats["first_in_valid_cycles"]
        pad_stats["module_pipeline_depth_cycles"] = pad_stats["first_out_valid_cycles"] - layer_hw_stats["first_in_valid_cycles"]
        layer_hw_stats['pad'] = pad_stats

        sliding_window_in_valid_signals = [signal for signal in self.signals if f"ConvolutionBlockFixed.sliding_window_io_in_valid" in signal][0]
        sliding_window_out_valid_signals = [signal for signal in self.signals if f"ConvolutionBlockFixed.sliding_window_io_out_valid" in signal][0]
        sliding_window_stats = {
            "first_in_valid_cycles": self.get_signal_first_edge(self.data[self.reference_signals[sliding_window_in_valid_signals]]),
            "last_in_valid_cycles": self.get_signal_last_edge(self.data[self.reference_signals[sliding_window_in_valid_signals]]),
            "first_out_valid_cycles": self.get_signal_first_edge(self.data[self.reference_signals[sliding_window_out_valid_signals]]),
            "last_out_valid_cycles": self.get_signal_last_edge(self.data[self.reference_signals[sliding_window_out_valid_signals]])
        }
        sliding_window_stats["module_total_cycles"] = sliding_window_stats["last_out_valid_cycles"] - sliding_window_stats["first_in_valid_cycles"]
        sliding_window_stats["module_pipeline_depth_cycles"] = sliding_window_stats["first_out_valid_cycles"] - layer_hw_stats["first_in_valid_cycles"]
        layer_hw_stats['sliding_window'] = sliding_window_stats

        squeeze_in_valid_signals = [signal for signal in self.signals if f"ConvolutionBlockFixed.squeeze_io_in_valid" in signal][0]
        squeeze_out_valid_signals = [signal for signal in self.signals if f"ConvolutionBlockFixed.squeeze_io_out_valid" in signal][0]
        squeeze_stats = {
            "first_in_valid_cycles": self.get_signal_first_edge(self.data[self.reference_signals[squeeze_in_valid_signals]]),
            "last_in_valid_cycles": self.get_signal_last_edge(self.data[self.reference_signals[squeeze_in_valid_signals]]),
            "first_out_valid_cycles": self.get_signal_first_edge(self.data[self.reference_signals[squeeze_out_valid_signals]]),
            "last_out_valid_cycles": self.get_signal_last_edge(self.data[self.reference_signals[squeeze_out_valid_signals]])
        }
        squeeze_stats["module_total_cycles"] = squeeze_stats["last_out_valid_cycles"] - squeeze_stats["first_in_valid_cycles"]
        squeeze_stats["module_pipeline_depth_cycles"] = squeeze_stats["first_out_valid_cycles"] - layer_hw_stats["first_in_valid_cycles"]
        layer_hw_stats['squeeze'] = squeeze_stats

        fork_in_valid_signals = [signal for signal in self.signals if f"ConvolutionBlockFixed.fork__io_in_valid" in signal][0]
        fork_out_valid_signals = [signal for signal in self.signals if f"ConvolutionBlockFixed.fork__io_out_valid" in signal][0]
        fork_stats = {
            "first_in_valid_cycles": self.get_signal_first_edge(self.data[self.reference_signals[fork_in_valid_signals]]),
            "last_in_valid_cycles": self.get_signal_last_edge(self.data[self.reference_signals[fork_in_valid_signals]]),
            "first_out_valid_cycles": self.get_signal_first_edge(self.data[self.reference_signals[fork_out_valid_signals]]),
            "last_out_valid_cycles": self.get_signal_last_edge(self.data[self.reference_signals[fork_out_valid_signals]])
        }
        fork_stats["module_total_cycles"] = fork_stats["last_out_valid_cycles"] - fork_stats["first_in_valid_cycles"]
        fork_stats["module_pipeline_depth_cycles"] = fork_stats["first_out_valid_cycles"] - layer_hw_stats["first_in_valid_cycles"]
        layer_hw_stats['fork'] = fork_stats

        vector_dot_in_valid_signals = [signal for signal in self.signals if f"ConvolutionBlockFixed.vector_dot_io_in_valid" in signal][0]
        vector_dot_out_valid_signals = [signal for signal in self.signals if f"ConvolutionBlockFixed.vector_dot_io_out_valid" in signal][0]
        vector_dot_stats = {
            "first_in_valid_cycles": self.get_signal_first_edge(self.data[self.reference_signals[vector_dot_in_valid_signals]]),
            "last_in_valid_cycles": self.get_signal_last_edge(self.data[self.reference_signals[vector_dot_in_valid_signals]]),
            "first_out_valid_cycles": self.get_signal_first_edge(self.data[self.reference_signals[vector_dot_out_valid_signals]]),
            "last_out_valid_cycles": self.get_signal_last_edge(self.data[self.reference_signals[vector_dot_out_valid_signals]])
        }
        vector_dot_stats["module_total_cycles"] = vector_dot_stats["last_out_valid_cycles"] - vector_dot_stats["first_in_valid_cycles"]
        vector_dot_stats["module_pipeline_depth_cycles"] = vector_dot_stats["first_out_valid_cycles"] - layer_hw_stats["first_in_valid_cycles"]
        layer_hw_stats['vector_dot'] = vector_dot_stats

        accum_in_valid_signals = [signal for signal in self.signals if f"ConvolutionBlockFixed.accum_io_in_valid" in signal][0]
        accum_out_valid_signals = [signal for signal in self.signals if f"ConvolutionBlockFixed.accum_io_out_valid" in signal][0]
        accum_stats = {
            "first_in_valid_cycles": self.get_signal_first_edge(self.data[self.reference_signals[accum_in_valid_signals]]),
            "last_in_valid_cycles": self.get_signal_last_edge(self.data[self.reference_signals[accum_in_valid_signals]]),
            "first_out_valid_cycles": self.get_signal_first_edge(self.data[self.reference_signals[accum_out_valid_signals]]),
            "last_out_valid_cycles": self.get_signal_last_edge(self.data[self.reference_signals[accum_out_valid_signals]])
        }
        accum_stats["module_total_cycles"] = accum_stats["last_out_valid_cycles"] - accum_stats["first_in_valid_cycles"]
        accum_stats["module_pipeline_depth_cycles"] = accum_stats["first_out_valid_cycles"] - layer_hw_stats["first_in_valid_cycles"]
        layer_hw_stats['accum'] = accum_stats

        glue_in_valid_signals = [signal for signal in self.signals if f"ConvolutionBlockFixed.glue_io_in_valid" in signal][0]
        glue_out_valid_signals = [signal for signal in self.signals if f"ConvolutionBlockFixed.glue_io_out_valid" in signal][0]
        glue_stats = {
            "first_in_valid_cycles": self.get_signal_first_edge(self.data[self.reference_signals[glue_in_valid_signals]]),
            "last_in_valid_cycles": self.get_signal_last_edge(self.data[self.reference_signals[glue_in_valid_signals]]),
            "first_out_valid_cycles": self.get_signal_first_edge(self.data[self.reference_signals[glue_out_valid_signals]]),
            "last_out_valid_cycles": self.get_signal_last_edge(self.data[self.reference_signals[glue_out_valid_signals]])
        }
        glue_stats["module_total_cycles"] = glue_stats["last_out_valid_cycles"] - glue_stats["first_in_valid_cycles"]
        glue_stats["module_pipeline_depth_cycles"] = glue_stats["first_out_valid_cycles"] - layer_hw_stats["first_in_valid_cycles"]
        layer_hw_stats['glue'] = glue_stats

        bias_in_valid_signals = [signal for signal in self.signals if f"ConvolutionBlockFixed.bias_io_in_valid" in signal][0]
        bias_out_valid_signals = [signal for signal in self.signals if f"ConvolutionBlockFixed.bias_io_out_valid" in signal][0]
        bias_stats = {
            "first_in_valid_cycles": self.get_signal_first_edge(self.data[self.reference_signals[bias_in_valid_signals]]),
            "last_in_valid_cycles": self.get_signal_last_edge(self.data[self.reference_signals[bias_in_valid_signals]]),
            "first_out_valid_cycles": self.get_signal_first_edge(self.data[self.reference_signals[bias_out_valid_signals]]),
            "last_out_valid_cycles": self.get_signal_last_edge(self.data[self.reference_signals[bias_out_valid_signals]])
        }
        bias_stats["module_total_cycles"] = bias_stats["last_out_valid_cycles"] - bias_stats["first_in_valid_cycles"]
        bias_stats["module_pipeline_depth_cycles"] = bias_stats["first_out_valid_cycles"] - layer_hw_stats["first_in_valid_cycles"]
        layer_hw_stats['bias'] = bias_stats

    def get_signals_per_module(self, module_name):
        if "Pad" in module_name:
            module_name = "PadBlockFixedDUT"
            in_valid_signals = [signal for signal in self.signals if f"{module_name}.io_in_0_valid" in signal]
            out_valid_signals = [signal for signal in self.signals if f"{module_name}.io_out_0_valid" in signal]
        elif "SlidingWindow" in module_name:
            module_name = "SlidingWindowBlockFixedDUT"
            in_valid_signals = [signal for signal in self.signals if f"{module_name}.io_in_0_valid" in signal]
            out_valid_signals = [signal for signal in self.signals if f"{module_name}.io_out_0_0_0_valid" in signal]
        elif "Squeeze" in module_name:
            module_name = "SqueezeBlockFixedDUT"
            in_valid_signals = [signal for signal in self.signals if f"{module_name}.io_in_0_0_valid" in signal]
            out_valid_signals = [signal for signal in self.signals if f"{module_name}.io_out_0_0_valid" in signal]
        elif "Fork" in module_name:
            module_name = "ForkBlockFixed"
        elif "Accum" in module_name:
            module_name = "AccumBlockFixedDUT"
            in_valid_signals = [signal for signal in self.signals if f"{module_name}.io_in_0_valid" in signal]
            out_valid_signals = [signal for signal in self.signals if f"{module_name}.io_out_0_valid" in signal]
        elif "Glue" in module_name:
            module_name = "GlueBlockFixed"
            raise NotImplementedError("GlueBlockFixed not implemented yet")
        elif "Bias" in module_name:
            module_name = "BiasBlockFixedDUT"
            in_valid_signals = [signal for signal in self.signals if f"{module_name}.io_in_0_valid" in signal]
            out_valid_signals = [signal for signal in self.signals if f"{module_name}.io_out_0_valid" in signal]
        elif "VectorDot" in module_name:
            module_name = "VectorDotBlockFixedDUT"
            in_valid_signals = [signal for signal in self.signals if f"{module_name}.io_in_0_0_valid" in signal]
            out_valid_signals = [signal for signal in self.signals if f"{module_name}.io_out_0_valid" in signal]


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