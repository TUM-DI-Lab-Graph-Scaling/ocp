import csv
import datetime
import itertools
import time
from abc import abstractmethod
from functools import reduce
from threading import Thread
from typing import List

import psutil
import torch.cuda
from pynvml.smi import nvidia_smi


class ResourceMonitor:
    @abstractmethod
    def headers(self) -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def measure(self) -> List[float]:
        raise NotImplementedError()


class CPUMemoryMonitor(ResourceMonitor):
    def headers(self) -> List[str]:
        return ["memory_used", "memory_free"]

    def measure(self) -> List[float]:
        mem = psutil.virtual_memory()
        return [mem.used, mem.free]


class GPUMemoryMonitor(ResourceMonitor):
    def __init__(self):
        self.nvsmi = nvidia_smi.getInstance()

    def headers(self) -> List[str]:
        return list(
            itertools.chain.from_iterable(
                [
                    [f"gpu_{i}_memory_used", f"gpu_{i}_memory_free"]
                    for i in range(torch.cuda.device_count())
                ]
            )
        )

    def measure(self) -> List[float]:
        gpu_stats = []
        for gpu in self.nvsmi.DeviceQuery("memory.used,memory.free")["gpu"]:
            gpu_stats.append(gpu["fb_memory_usage"]["used"])
            gpu_stats.append(gpu["fb_memory_usage"]["free"])
        return gpu_stats


class ResourceMonitorThread(Thread):
    def __init__(self, delay, resource_file):
        super(ResourceMonitorThread, self).__init__()
        self.resource_file = resource_file
        self.resource_writer = csv.writer(self.resource_file, delimiter=",")
        self.stop = False
        self.delay = delay
        self.epoch = 0
        self.log_results = False
        self.monitors: List[ResourceMonitor] = []

    def run(self) -> None:
        headers = ["datetime", "epoch"]
        headers.extend(
            reduce(list.__add__, [m.headers() for m in self.monitors])
        )
        self.resource_writer.writerow(headers)
        self.resource_file.flush()

        while not self.stop:
            if self.log_results:
                self.eval_monitors()
            time.sleep(self.delay)

    def add_monitor(self, monitor: ResourceMonitor):
        self.monitors.append(monitor)

    def eval_monitors(self):
        row = [datetime.datetime.now(), self.epoch]
        row.extend(reduce(list.__add__, [m.measure() for m in self.monitors]))
        self.resource_writer.writerow(row)
        self.resource_file.flush()
