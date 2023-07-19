import unittest

import numpy as np

from mjregrasping.homotopy_utils import get_h_signature, make_ring_skeleton


class TestHomotopyUtils(unittest.TestCase):
    def test_non_closed_h_signature(self):
        path = np.array([[0, 0, 0], [1, 0, 0]])
        skeletons = {'obstacle': make_ring_skeleton(np.zeros(3), np.array([1, 0, 0]), 0.5)}
        with self.assertRaises(ValueError):
            get_h_signature(path, skeletons)


if __name__ == '__main__':
    unittest.main()
