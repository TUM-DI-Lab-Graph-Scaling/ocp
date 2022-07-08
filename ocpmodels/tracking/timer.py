import time


class Timer:
    def __init__(self):
        self._time = 0
        self._start = None

    def start(self):
        self._start = time.perf_counter()

    def stop(self):
        if self._start is None:
            raise Exception("Timer was never started.")

        elapsed_time = time.perf_counter() - self._start
        self._time += elapsed_time
        self._start = None

    def elapsed(self):
        return self._time

    def reset(self):
        self._time = 0
        self._start = None
