from pathlib import Path
import tempfile
import unittest
import torch
from PIL import Image
from training_cache import CachedCollator


class CacheTests(unittest.TestCase):
    def test_exact_tensors_and_invalidation(self):
        calls=[]
        def collate(rows):
            calls.append(1)
            return {'input_ids':torch.tensor([[1,2,3]]), 'labels':torch.tensor([[-100,-100,rows[0]['label']]]),
                    'pixel_values':torch.arange(12,dtype=torch.float32).view(1,3,2,2)}
        with tempfile.TemporaryDirectory() as td:
            cache=CachedCollator(collate,td,'processor-v1')
            rows=[{'label':3,'image':Image.new('RGB',(2,2),'red')}]
            expected=cache(rows);actual=cache(rows)
            for key in expected:self.assertTrue(torch.equal(expected[key],actual[key]))
            self.assertEqual(len(calls),1)
            actual['labels'].fill_(-1)
            self.assertEqual(cache(rows)['labels'][0,-1].item(),3)
            rows[0]['label']=4;self.assertEqual(cache(rows)['labels'][0,-1].item(),4)
            rows[0]['image']=Image.new('RGB',(2,2),'blue');cache(rows)
            CachedCollator(collate,td,'processor-v2')(rows)
            self.assertEqual(len(calls),4)

    def test_budget_and_corruption_fall_back_to_processing(self):
        def collate(_):return {'input_ids':torch.tensor([[1,2]])}
        with tempfile.TemporaryDirectory() as td:
            tiny=CachedCollator(collate,td,'v1',max_gb=1e-8)
            tiny([1]);self.assertFalse(list(Path(td).glob('*.pt')))
            self.assertEqual(tiny.stats['skipped_writes'],1)
            cache=CachedCollator(collate,td,'v1');cache([1])
            next(Path(td).glob('*.pt')).write_bytes(b'invalid')
            self.assertTrue(torch.equal(cache([1])['input_ids'],torch.tensor([[1,2]])))
            self.assertEqual(cache.stats['corrupt_entries'],1)


if __name__=='__main__':unittest.main()
