import csv
import time
from pathlib import Path

from ocpmodels.common import distutils
from ocpmodels.tracking.timer import Timer


class MetricsWriter:
    def __init__(self, metrics_path, model_name):
        self.path = (
            Path(metrics_path) / model_name / (str(int(time.time())) + ".csv")
        )
        self.file = open(self.path, "a")
        self.writer = csv.writer(self.file, delimiter=",")
        self.epoch_timer = Timer()
        self.forward_timer = Timer()
        self.backward_timer = Timer()

    def write_stats(self):
        distutils.synchronize()
        if distutils.is_master():
            self.writer.writerow(
                [
                    self.epoch_timer.elapsed(),
                    self.forward_timer.elapsed(),
                    self.backward_timer.elapsed(),
                ]
            )
            self.file.flush()

            self.epoch_timer.reset()
            self.forward_timer.reset()
            self.backward_timer.reset()

    def start_epoch(self):
        self.epoch_timer.start()

    def end_epoch(self):
        self.epoch_timer.stop()
        self.write_stats()

    def start_forward(self):
        self.forward_timer.start()

    def end_forward(self):
        self.forward_timer.stop()

    def start_backward(self):
        self.backward_timer.start()

    def end_backward(self):
        self.backward_timer.stop()

    def close(self):
        self.file.close()
