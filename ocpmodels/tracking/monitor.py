import statistics
import time
from abc import abstractmethod
from functools import reduce
from threading import Thread

import psutil
from pynvml.smi import nvidia_smi


class ResourceMonitor:
    def __init__(self):
        self.resource_stats = None
        self.reset()

    @abstractmethod
    def measure(self):
        raise NotImplementedError()

    def eval_stats(self):
        return [
            statistics.fmean(signal_column)
            for signal_column in self.resource_stats
        ]

    @abstractmethod
    def reset(self):
        raise NotImplementedError()


class CPUMemoryMonitor(ResourceMonitor):
    def measure(self):
        self.resource_stats[0].append(psutil.virtual_memory()[3])

    def reset(self):
        self.resource_stats = [[]]


class GPUMemoryMonitor(ResourceMonitor):
    def __init__(self, num_gpus):
        self.nvsmi = nvidia_smi.getInstance()
        self.num_gpus = len(self.nvsmi.DeviceQuery("name")["gpu"])
        super().__init__()

    def measure(self):
        for i, gpu in enumerate(
            self.nvsmi.DeviceQuery("memory.used,memory.free")["gpu"]
        ):
            self.resource_stats[i * 2].append(gpu["fb_memory_usage"]["used"])
            self.resource_stats[i * 2 + 1].append(
                gpu["fb_memory_usage"]["free"]
            )

    def reset(self):
        self.resource_stats = [[] for _ in range(self.num_gpus * 2)]


class ResourceMonitorThread(Thread):
    def __init__(self, delay):
        super(ResourceMonitorThread, self).__init__()
        self.stopped = False
        self.delay = delay
        self.monitors = []

    def run(self) -> None:
        while not self.stopped:
            for m in self.monitors:
                m.measure()
            time.sleep(self.delay)

    def add_monitor(self, monitor: ResourceMonitor):
        self.monitors.append(monitor)

    def eval_monitors(self):
        return reduce(list.__add__, [m.eval_stats() for m in self.monitors])

    def stop(self):
        self.stopped = True

    def reset(self):
        self.stop()
        for m in self.monitors:
            m.reset()
