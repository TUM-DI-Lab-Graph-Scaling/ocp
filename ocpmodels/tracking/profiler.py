import csv
import time
from enum import Enum
from pathlib import Path
from typing import Optional

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
    def __init__(self, config, trainer, model_name):
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
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.id = int(time.time())
        self.runtime_path = self.dir_path / (str(self.id) + "_runtimes.csv")
        self.resource_path = self.dir_path / (str(self.id) + "_resources.csv")
        self.current_epoch = 0
        self.epoch_timer = Timer()
        self.phase_timers = {
            Phase.DATALOADING: Timer(),
            Phase.FORWARD: Timer(),
            Phase.BACKWARD: Timer(),
        }

    def __enter__(self):
        self.runtime_file = open(self.runtime_path, "a")
        self.resource_file = open(self.resource_path, "a")

        self.runtime_writer = csv.writer(self.runtime_file, delimiter=",")

        if distutils.is_master():
            self.resource_monitor_thread = ResourceMonitorThread(
                self.config["resource_poll_interval"], self.resource_file
            )
            self.resource_monitor_thread.add_monitor(CPUMemoryMonitor())
            self.resource_monitor_thread.add_monitor(GPUMemoryMonitor())
            self.resource_monitor_thread.start()
        return self

    def write_stats(self):
        row = [
            distutils.get_rank(),
            self.current_epoch,
            self.epoch_timer.elapsed(),
        ]
        row.extend([t.elapsed() for t in self.phase_timers.values()])
        self.runtime_writer.writerow(row)
        self.runtime_file.flush()

        self.epoch_timer.reset()
        for t in self.phase_timers.values():
            t.reset()

    def start_epoch(self):
        if distutils.is_master():
            self.resource_monitor_thread.epoch = self.current_epoch
            self.resource_monitor_thread.log_results = True
        self.epoch_timer.start()

    def end_epoch(self):
        self.epoch_timer.stop()
        if distutils.is_master():
            self.resource_monitor_thread.log_results = False
        self.write_stats()
        self.current_epoch += 1

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
