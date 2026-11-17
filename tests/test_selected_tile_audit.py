import unittest

import numpy as np

from data_preprocessing.evaluate_selected_tiles import selected_indices


class SelectedTileAuditTests(unittest.TestCase):
    def setUp(self):
        self.coords = np.array([[10 * i, 20 * i] for i in range(20)])
        self.rows = [dict(position=i, x=10 * (19 - i), y=20 * (19 - i)) for i in range(16)]

    def test_matches_coordinates_in_montage_order_not_cached_order(self):
        indices, ordered = selected_indices(self.coords, list(reversed(self.rows)))
        self.assertEqual(indices.tolist(), list(range(19, 3, -1)))
        self.assertEqual([r['position'] for r in ordered], list(range(16)))

    def test_rejects_missing_or_ambiguous_coordinates(self):
        with self.assertRaises(ValueError):
            selected_indices(self.coords[:-1], self.rows)
        with self.assertRaises(ValueError):
            selected_indices(np.concatenate([self.coords, self.coords[-1:]]), self.rows)

    def test_rejects_missing_positions_or_duplicate_tiles(self):
        with self.assertRaises(ValueError):
            selected_indices(self.coords, self.rows[:-1])
        rows = [dict(r) for r in self.rows]
        rows[1].update(x=rows[0]['x'], y=rows[0]['y'])
        with self.assertRaises(ValueError):
            selected_indices(self.coords, rows)


if __name__ == '__main__':
    unittest.main()
