import csv
import time
from enum import Enum
from pathlib import Path

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
    def __init__(self, metrics_path, model_name):
        self.dir_path = Path(metrics_path) / model_name
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.id = int(time.time())
        self.runtime_path = self.dir_path / (str(id) + "runtimes.csv")
        self.resource_path = self.dir_path / (str(id) + "resources.csv")
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
                20, self.resource_file
            )
            self.resource_monitor_thread.add_monitor(CPUMemoryMonitor())
            self.resource_monitor_thread.add_monitor(GPUMemoryMonitor())
            self.resource_monitor_thread.start()
        return self

    def write_stats(self):
        row = [distutils.get_rank(), self.epoch_timer.elapsed()]
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
