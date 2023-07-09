from abc import ABC

import numpy as np
import pysdf_tools
from pymjregrasping_cpp import check_first_order_homotopy_points

from mjregrasping.magnetic_fields import get_true_h_signature


class PathComparer(ABC):

    def get_signature(self, path):
        raise NotImplementedError()

    def check_equal(self, h1, h2):
        raise NotImplementedError()


class TrueHomotopyComparer(PathComparer):

    def __init__(self, skeletons):
        self.skeletons = skeletons

    def get_signature(self, path):
        return get_true_h_signature(path, self.skeletons)

    def check_equal(self, h1, h2):
        if np.shape(h1) != np.shape(h2):
            return False
        try:
            return np.all(h1 == h2)
        except ValueError:
            return np.allclose(np.sort(h1, axis=0), np.sort(h2, axis=0))


class FirstOrderComparer(PathComparer):

    def __init__(self, sdf: pysdf_tools.SignedDistanceField):
        self.sdf = sdf

    def get_signature(self, path):
        return path

    def check_equal(self, h1, h2):
        # Here the signatures are just the paths themselves
        is_equal = check_first_order_homotopy_points(self.sdf, h1, h2)
        return is_equal
