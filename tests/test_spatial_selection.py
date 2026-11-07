import unittest
import numpy as np
from data_preprocessing.tile_selection import select_tiles, spatial_metrics

class SpatialSelectionTests(unittest.TestCase):
    def test_separation_and_attention_order_with_fixed_candidates(self):
        xy=np.array([[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]])*100
        w=np.arange(6,0,-1)
        top,_=select_tiles(w,xy,3,tile_size=100)
        self.assertEqual(top.tolist(),[0,1,2])
        chosen,report=select_tiles(w,xy,3,method='spatial',tile_size=100,candidate_pool=5)
        self.assertEqual(chosen.tolist(),[0,2,4])
        self.assertEqual(report['relaxed_count'],0)
        self.assertGreaterEqual(spatial_metrics(xy,chosen,100)['min_distance_tiles'],2)

    def test_infeasible_distance_is_explicit_and_deterministic(self):
        xy=np.array([[0,0],[1,0],[2,0],[3,0]])
        kwargs=dict(method='spatial',tile_size=100,candidate_pool=4)
        a,r=select_tiles([4,3,2,1],xy,4,**kwargs)
        self.assertEqual(a.tolist(),[0,1,2,3])
        self.assertEqual(r['zero_distance_fallback_count'],3)
        self.assertEqual(len(set(a)),4)

    def test_no_out_of_pool_selection_or_coordinate_mismatch(self):
        xy=np.array([[0,0],[1,0],[2,0],[1000,0]])
        chosen,_=select_tiles([4,3,2,1],xy,2,method='spatial',candidate_pool=3,tile_size=1)
        self.assertNotIn(3,chosen)
        with self.assertRaises(ValueError): select_tiles([1,2],[[0,0]],2)
        with self.assertRaises(ValueError): select_tiles([1,2],[[0,0],[0,0]],2)

    def test_adjacent_chain_is_one_component(self):
        m=spatial_metrics(np.array([[0,0],[1,0],[2,0],[10,0]]),np.arange(4),1)
        self.assertEqual(m['largest_adjacent_component'],3)

if __name__=='__main__':unittest.main()
