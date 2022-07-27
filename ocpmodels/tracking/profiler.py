import csv
import datetime
import json
import shutil
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import torch.distributed

from ocpmodels.common import distutils
from ocpmodels.tracking.monitor import (
    CPUMemoryMonitor,
    GPUMemoryMonitor,
    ResourceMonitorThread,
)
from ocpmodels.tracking.timer import Timer


class Phase(Enum):
    DATALOADING = 1
    FORWARD = 2
    BACKWARD = 3


class Profiler:
    def __init__(self, config, trainer, model_name, deepspeed_config):
        assert (
            "metrics_path" in config
        ), "Profiler config contains no metrics path."
        assert (
            "resource_poll_interval" in config
        ), "Resource utilization poll interval must be defined."

        self.config = config
        task_name = "is2re" if trainer.__name__ == "EnergyTrainer" else "s2ef"
        self.dir_path = (
            Path(self.config["metrics_path"]) / task_name / model_name
        )
        self.deepspeed_config = deepspeed_config
        self.epoch = 0
        self.epoch_timer = Timer()
        self.phase_timers = {
            Phase.DATALOADING: Timer(),
            Phase.FORWARD: Timer(),
            Phase.BACKWARD: Timer(),
        }

    def __enter__(self):
        self.dir_path.mkdir(parents=True, exist_ok=True)
        if distutils.is_master():
            ids = [int(time.time())]
        else:
            ids = [None]
        torch.distributed.broadcast_object_list(ids)

        self.id = ids[0]
        self.ds_stage = 0
        additional_flags = []
        if self.deepspeed_config:
            with open(self.deepspeed_config) as ds_f:
                ds_config = json.load(ds_f)
                self.ds_stage = ds_config["zero_optimization"]["stage"]

                if "offload_optimizer" in ds_config["zero_optimization"]:
                    offloading_device = ds_config["zero_optimization"][
                        "offload_optimizer"
                    ]["device"]
                    additional_flags.append(
                        f"_offloadOptimizer[{offloading_device}]"
                    )

                if "offload_param" in ds_config["zero_optimization"]:
                    offloading_device = ds_config["zero_optimization"][
                        "offload_param"
                    ]["device"]
                    additional_flags.append(
                        f"_offloadParam[{offloading_device}]"
                    )

        base_path = (
            self.dir_path
            / f"{str(self.id)}_stage{self.ds_stage}_{distutils.get_world_size()}gpus{''.join(additional_flags)}"
        )
        self.runtime_path = Path(str(base_path) + "_runtimes.csv")
        self.resource_path = Path(str(base_path) + "_resources.csv")
        self.torch_cuda_path = Path(str(base_path) + "_torch_cuda.csv")

        if self.deepspeed_config is not None and distutils.is_master():
            self.deepspeed_config_path = self.dir_path / (
                str(self.id) + "_ds_config.json"
            )
            shutil.copyfile(self.deepspeed_config, self.deepspeed_config_path)

        self.runtime_file = open(self.runtime_path, "a")
        self.resource_file = open(self.resource_path, "a")
        self.torch_cuda_file = open(self.torch_cuda_path, "a")

        self.runtime_writer = csv.writer(self.runtime_file, delimiter=",")
        self.torch_cuda_writer = csv.writer(
            self.torch_cuda_file, delimiter=","
        )

        if distutils.is_master():
            self.write_runtime_headers()
            self.write_torch_cuda_headers()

            self.resource_monitor_thread = ResourceMonitorThread(
                self.config["resource_poll_interval"], self.resource_file
            )
            self.resource_monitor_thread.add_monitor(CPUMemoryMonitor())
            self.resource_monitor_thread.add_monitor(GPUMemoryMonitor())
            self.resource_monitor_thread.start()

        return self

    def write_runtime_headers(self):
        self.runtime_writer.writerow(
            [
                "rank",
                "epoch",
                "epoch_time",
                "dataloading_time",
                "forward_time",
                "backward_time",
            ]
        )
        self.runtime_file.flush()

    def write_torch_cuda_headers(self):
        self.torch_cuda_writer.writerow(
            [
                "datetime",
                "epoch",
                "rank",
                "gpu_memory_allocated",
                "gpu_memory_reserved",
            ]
        )
        self.runtime_file.flush()

    def write_runtime_stats(self):
        row = [
            distutils.get_rank(),
            self.epoch,
            self.epoch_timer.elapsed(),
        ]
        row.extend([t.elapsed() for t in self.phase_timers.values()])
        self.runtime_writer.writerow(row)
        self.runtime_file.flush()

        self.epoch_timer.reset()
        for t in self.phase_timers.values():
            t.reset()

    def write_torch_cuda_stats(self):
        row = [
            datetime.datetime.now(),
            self.epoch,
            distutils.get_rank(),
            torch.cuda.memory_allocated(),
            torch.cuda.memory_reserved(),
        ]

        self.torch_cuda_writer.writerow(row)
        self.torch_cuda_file.flush()

    def record_gpu_memory(self):
        self.write_torch_cuda_stats()

    def start_epoch(self):
        if distutils.is_master():
            self.resource_monitor_thread.epoch = self.epoch
            self.resource_monitor_thread.log_results = True
        self.epoch_timer.start()

    def end_epoch(self):
        self.epoch_timer.stop()
        if distutils.is_master():
            self.resource_monitor_thread.log_results = False
        self.write_runtime_stats()
        self.epoch += 1

    def start_phase(self, phase):
        self.phase_timers[phase].start()

    def end_phase(self, phase):
        self.phase_timers[phase].stop()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.runtime_file.close()
        self.resource_file.close()
        if distutils.is_master():
            self.resource_monitor_thread.stop = True
            self.resource_monitor_thread.join(20)


def profiler_phase(phase: Phase):
    def inner(func):
        def phase_wrapper(*args, **kwargs):
            global profiler
            if profiler is not None:
                profiler.start_phase(phase)
            return_value = func(*args, **kwargs)
            if profiler is not None:
                profiler.end_phase(phase)
            return return_value

        return phase_wrapper

    return inner


profiler: Optional[Profiler] = None


def set_profiler(p: Profiler):
    global profiler
    profiler = p
