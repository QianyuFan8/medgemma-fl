"""Top-attention rule for the 16-tile MedGemma montage."""
import unittest

import numpy as np

from data_preprocessing.abmil_montage import select_top_tiles


class TileSelectionTests(unittest.TestCase):
    def test_highest_attention_ignores_location(self):
        weights = np.array([0.9, 0.8, 0.7, 0.1])
        got = select_top_tiles(weights, 3)
        self.assertEqual(list(got), [0, 1, 2])

    def test_order_follows_attention(self):
        weights = np.array([0.2, 0.9, 0.5])
        got = select_top_tiles(weights, 3)
        self.assertEqual(list(got), [1, 2, 0])

    def test_k_caps_the_selection(self):
        weights = np.array([0.1, 0.4, 0.3, 0.2])
        self.assertEqual(list(select_top_tiles(weights, 16)), [1, 2, 3, 0])
        self.assertEqual(list(select_top_tiles(weights, 0)), [])


if __name__ == "__main__":
    unittest.main()
