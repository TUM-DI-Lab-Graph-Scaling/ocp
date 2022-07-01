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
        self.path = self.dir_path / (str(int(time.time())) + ".csv")
        self.file = open(self.path, "a")
        self.writer = csv.writer(self.file, delimiter=",")
        self.epoch_timer = Timer()
        self.phase_timers = {
            Phase.DATALOADING: Timer(),
            Phase.FORWARD: Timer(),
            Phase.BACKWARD: Timer(),
        }

        # self.resource_monitor_thread = ResourceMonitorThread(1)
        # self.resource_monitor_thread.add_monitor(CPUMemoryMonitor())
        # self.resource_monitor_thread.add_monitor(GPUMemoryMonitor(distutils.get_world_size()))
        # self.resource_monitor_thread.start()

    def write_stats(self):
        # monitor_stats = self.resource_monitor_thread.eval_monitors()
        row = [distutils.get_rank(), self.epoch_timer.elapsed()]
        row.extend([t.elapsed() for t in self.phase_timers.values()])
        # row.extend(monitor_stats)
        self.writer.writerow(row)
        self.file.flush()

        self.epoch_timer.reset()
        for t in self.phase_timers.values():
            t.reset()

    def start_epoch(self):
        # self.resource_monitor_thread.reset()
        self.epoch_timer.start()

    def end_epoch(self):
        self.epoch_timer.stop()
        self.write_stats()

    def start_phase(self, phase):
        self.phase_timers[phase].start()

    def end_phase(self, phase):
        self.phase_timers[phase].stop()

    def close(self):
        self.file.close()
