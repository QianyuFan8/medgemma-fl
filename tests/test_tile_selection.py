"""Spacing rule for the 16-tile MedGemma montage."""
import unittest

import numpy as np

from data_preprocessing.abmil_montage import select_spaced_tiles


class TileSelectionTests(unittest.TestCase):
    def test_spacing_then_fill(self):
        coords = np.array([[0, 0], [10, 0], [20, 0], [30, 0]])
        weights = np.array([0.9, 0.8, 0.7, 0.1])
        got = select_spaced_tiles(weights, coords, 3, 20)
        self.assertEqual(list(got), [0, 2, 1])

    def test_spaced_tiles_stay_in_attention_order(self):
        coords = np.array([[0, 0], [40, 0], [0, 40]])
        weights = np.array([0.2, 0.9, 0.5])
        got = select_spaced_tiles(weights, coords, 3, 20)
        self.assertEqual(list(got), [1, 2, 0])


if __name__ == "__main__":
    unittest.main()
