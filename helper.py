import time


class Timer:
    def __init__(self, name):
        self.name = name
        self.total = 0
        self.last = None

    def start(self):
        self.last = time.time()

    def stop(self):
        if self.last is not None:
            self.total += time.time() - self.last
        self.last = None

    def __str__(self):
        return f'Timer {self.name}: Total Time {self.total}'


class Counter:
    def __init__(self, name):
        self.name = name
        self.total = 0

    def add(self, x):
        self.total += x

    def __str__(self):
        return f'Counter {self.name}: {self.total}'
