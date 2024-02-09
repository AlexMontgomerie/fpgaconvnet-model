from dataclasses import dataclass
from vcdvcd import VCDVCD

@dataclass
class VCDWaveformParser:
    vcd_path: str
    is_partition: bool = False
    offset: int = 0
    MAXPOOLING_COUNTER: int = 0
    GLOBALPOOLING_COUNTER: int = 0
    SQUEEZE_COUNTER: int = 0

    def __post_init__(self):
        self.vcd = VCDVCD(self.vcd_path)
        self.signals = self.vcd.signals
        self.reference_signals = self.vcd.references_to_ids
        self.data = self.vcd.data
        if self.is_partition:
            offset = [signal for signal in self.signals if "partition.io_in_0_0_valid" in signal]
            assert len(offset) == 1, "Could not find offset signal."
            self.offset = self.get_signal_first_edge(self.data[self.reference_signals[offset[0]]]) + 1

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

    def calculate_fire_signals(self, layer_hw_stats, clock):
        clock_signal = self.data[self.reference_signals[clock]]
        in_valid_signal = self.data[self.reference_signals[layer_hw_stats["in_valid_signal"]]]
        in_ready_signal = self.data[self.reference_signals[layer_hw_stats["in_ready_signal"]]]
        out_valid_signal = self.data[self.reference_signals[layer_hw_stats["out_valid_signal"]]]
        out_ready_signal = self.data[self.reference_signals[layer_hw_stats["out_ready_signal"]]]

        in_fire_signal = [(clock_time, '1') if in_valid_signal[clock_time] == '1' and in_ready_signal[clock_time] == '1' else (clock_time, '0') for (clock_time, _) in clock_signal.tv]
        out_fire_signal = [(clock_time, '1') if out_valid_signal[clock_time] == '1' and out_ready_signal[clock_time] == '1' else (clock_time, '0') for (clock_time, _) in clock_signal.tv]

        return in_fire_signal, out_fire_signal

    def get_start_depth(self, fire_signal, start_cycle, end_cycle):
        return sum(1 for (cycle, value) in fire_signal if start_cycle <= cycle <= end_cycle and value == '1') // 2

    def get_rate(self, fire_signal, start_cycle, end_cycle):
        word_count = self.get_start_depth(fire_signal, start_cycle, end_cycle)
        total_cycles = (end_cycle - start_cycle) // 2

        return word_count / total_cycles

    def get_signals_per_layer(self, layer_name):
        if ("GlobalAveragePool" in layer_name or "GlobalMaxPool" in layer_name ) and not "squeeze" in layer_name:
            if self.GLOBALPOOLING_COUNTER == 0:
                layer_name = "GlobalAveragePoolingFixed"
            else:
                layer_name = f"GlobalAveragePoolingFixed_{self.GLOBALPOOLING_COUNTER}"
            self.GLOBALPOOLING_COUNTER += 1
        elif "MaxPool" in layer_name and not "squeeze" in layer_name:
            if self.MAXPOOLING_COUNTER == 0:
                layer_name = "PoolingBlockFixed"
            else:
                layer_name = f"PoolingBlockFixed_{self.MAXPOOLING_COUNTER}"
            self.MAXPOOLING_COUNTER += 1
        elif "Convolution" in layer_name and not "squeeze" in layer_name:
            layer_name = "ConvolutionBlockFixed"
        elif "squeeze" in layer_name.lower():
            if self.SQUEEZE_COUNTER == 0:
                layer_name = "SqueezeLayerFixed"
            else:
                layer_name = f"SqueezeLayerFixed_{self.SQUEEZE_COUNTER}"
            self.SQUEEZE_COUNTER += 1
        elif "Gemm" in layer_name:
            layer_name = "InnerProductBlockFixed"

        in_valid_signals = [signal for signal in self.signals if f"{layer_name}.io_in_0_0_valid" in signal]
        in_ready_signals = [signal for signal in self.signals if f"{layer_name}.io_in_0_0_ready" in signal]
        out_valid_signals = [signal for signal in self.signals if f"{layer_name}.io_out_0_0_valid" in signal]
        out_ready_signals = [signal for signal in self.signals if f"{layer_name}.io_out_0_0_ready" in signal]

        layer_hw_stats = {
            "in_valid_signal": in_valid_signals[0],
            "in_ready_signal": in_ready_signals[0],
            "out_valid_signal": out_valid_signals[0],
            "out_ready_signal": out_ready_signals[0]
        }
        return layer_hw_stats

    def get_layer_stats(self, layer_type):

        clock = [signal for signal in self.signals if "clock" in signal][0]

        layer_hw_stats = self.get_signals_per_layer(layer_type)
        in_fire_signal, out_fire_signal = self.calculate_fire_signals(layer_hw_stats, clock)

        layer_hw_stats["first_in_valid_cycles"] = self.get_signal_first_edge(self.data[self.reference_signals[layer_hw_stats["in_valid_signal"]]])
        layer_hw_stats["last_in_valid_cycles"] = self.get_signal_last_edge(self.data[self.reference_signals[layer_hw_stats["in_valid_signal"]]])
        layer_hw_stats["first_out_valid_cycles"] = self.get_signal_first_edge(self.data[self.reference_signals[layer_hw_stats["out_valid_signal"]]])
        layer_hw_stats["last_out_valid_cycles"] = self.get_signal_last_edge(self.data[self.reference_signals[layer_hw_stats["out_valid_signal"]]])
        layer_hw_stats["layer_total_cycles"] = layer_hw_stats["last_out_valid_cycles"] - layer_hw_stats["first_in_valid_cycles"]
        layer_hw_stats["layer_pipeline_depth_cycles"] = layer_hw_stats["first_out_valid_cycles"] - layer_hw_stats["first_in_valid_cycles"]
        layer_hw_stats["partition_pipeline_depth_cycles"] = layer_hw_stats["first_out_valid_cycles"] - self.offset
        layer_hw_stats["layer_start_depth"] = self.get_start_depth(fire_signal=in_fire_signal, start_cycle=layer_hw_stats["first_in_valid_cycles"]*2, end_cycle=layer_hw_stats["first_out_valid_cycles"]*2)
        layer_hw_stats["initial_rate_in_per_stream"] = self.get_rate(fire_signal=in_fire_signal, start_cycle=layer_hw_stats["first_in_valid_cycles"]*2, end_cycle=layer_hw_stats["first_out_valid_cycles"]*2)
        layer_hw_stats["average_rate_in_per_stream"] = self.get_rate(fire_signal=in_fire_signal, start_cycle=layer_hw_stats["first_in_valid_cycles"]*2, end_cycle=layer_hw_stats["last_in_valid_cycles"]*2)
        layer_hw_stats["average_rate_out_per_stream"] = self.get_rate(fire_signal=out_fire_signal, start_cycle=layer_hw_stats["first_out_valid_cycles"]*2, end_cycle=layer_hw_stats["last_out_valid_cycles"]*2)

        # if layer_type == "Convolution":
        #     self.get_conv_modules_stats(layer_hw_stats)

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
