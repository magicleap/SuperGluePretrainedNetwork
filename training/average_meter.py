import numpy as np


class AverageMeter:
    def __init__(self):
        self.reset()

    def add_value(self, value):
        self.values.append(value)

    def get_value(self, last_values=None):
        if last_values is None:
            return np.mean(self.values)
        return np.mean(self.values[-last_values:])

    def reset(self):
        self.values = []
