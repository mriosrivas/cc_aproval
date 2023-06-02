import numpy as np


class AuxiliaryFunctions:
    def __init__(self):
        pass

    def get_years(self, x):
        return np.round(np.abs(x / 365), 0).astype("int")
