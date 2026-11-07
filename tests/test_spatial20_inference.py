import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from run_spatial20_inference import (extend_selection, original_indices, prepare_pairs, prompt_for_twenty,
                                     save_prepared_manifest, load_prepared_pairs, validate_weight_scope)
from prepare_multiimage_diagnostics import convert_record
from subtype_utils import make_subtype_record


class SpatialTwentyTests(unittest.TestCase):
    def test_adapter_scope_is_verified(self):
        language = {'base_model.model.language_model.layers.0.q_proj.lora_A.default.weight': None}
        both = {**language, 'base_model.model.vision_tower.layers.0.q_proj.lora_B.default.weight': None}
        validate_weight_scope(language, 'language-only')
        validate_weight_scope(both, 'lora-only')
        with self.assertRaises(ValueError): validate_weight_scope(language, 'lora-only')
        with self.assertRaises(ValueError): validate_weight_scope(both, 'language-only')
        with self.assertRaises(ValueError):
            validate_weight_scope({**both, 'base_model.model.lm_head.weight': None}, 'lora-only')

    def test_preserves_original_order_and_appends_four_at_strict_distance(self):
        coords = np.array([(i * 448, 0) for i in range(25)])
        original = list(reversed(range(16)))
        chosen = extend_selection(np.arange(25)[::-1], coords, original, 224)
        self.assertEqual(chosen[:16], original)
        self.assertEqual(chosen[16:], [16, 17, 18, 19])
        for j in range(16, 20):
            self.assertGreaterEqual(np.linalg.norm(coords[chosen[:j]]-coords[chosen[j]], axis=1).min(), 448)

    def test_does_not_relax_spacing_or_leave_candidate_pool(self):
        coords = np.array([(i * 224, 0) for i in range(32)])
        original = list(range(0, 32, 2))
        with self.assertRaisesRegex(ValueError, 'No spacing relaxation'):
            extend_selection(np.arange(32)[::-1], coords, original, 224)
        coords = np.array([(i * 448, 0) for i in range(25)])
        with self.assertRaisesRegex(ValueError, 'candidate pool'):
            extend_selection(np.arange(25)[::-1], coords, list(range(16)), 224, 20, 5)

    def test_prompt_changes_only_image_count_not_cpg_text_or_instructions(self):
        row = make_subtype_record('r', 'POS', ['cg00001'], [.123456], ['NEG','POS'],
                                  'multimodal', 'images/x.png', 'site-1')
        old = convert_record(row, 'multimodal')['prompt']
        new = prompt_for_twenty(old)
        self.assertEqual(new.replace('20 separate', '16 separate').replace('all 20', 'all 16'), old)
        self.assertEqual(new.split('[DNA methylation]')[1], old.split('[DNA methylation]')[1])
        with self.assertRaises(ValueError): prompt_for_twenty(row['prompt'])

    def test_index_uses_saved_positions_and_rejects_duplicates(self):
        coords = np.array([(i * 448, 0) for i in range(20)])
        rows = [dict(position=str(i), x=str(448*i), y='0') for i in range(16)]
        chosen, _ = original_indices(coords, list(reversed(rows)))
        self.assertEqual(chosen, list(range(16)))
        rows[-1]['x'] = '0'
        with self.assertRaisesRegex(ValueError, 'Repeated'): original_indices(coords, rows)

    def test_preparation_reuses_source_pixels_and_changes_no_source_data(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td); data = base/'data'; data.mkdir()
            (data/'images').mkdir(); (base/'montage').mkdir(); (base/'emb').mkdir()
            (base/'abmil/attention').mkdir(parents=True); (base/'svs').mkdir(); (base/'output').mkdir()
            coords = np.array([(i*448, 0) for i in range(25)])
            weights = np.linspace(1, .01, 25)
            np.save(base/'abmil/attention/slide.npy', weights)
            (base/'emb/slide.h5').write_bytes(b'fixture'); (base/'svs/slide.svs').write_bytes(b'fixture')
            canvas = Image.new('RGB', (896,896))
            for i in range(16):canvas.paste((i,20,30), (i%4*224,i//4*224,(i%4+1)*224,(i//4+1)*224))
            canvas.save(data/'images/x.png')
            row = convert_record(make_subtype_record('val1','POS',['cg00001'],[.123456],['NEG','POS'],
                                      'multimodal','images/x.png','site-1'), 'multimodal')
            (data/'validation.json').write_text(json.dumps([row]))
            (data/'task.json').write_text('{}')
            with (data/'splits.tsv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['patient_id','canonical_uid','split','diagnosis','slide_id'],delimiter='\t')
                writer.writeheader();writer.writerow(dict(patient_id='val1',canonical_uid='u1',split='validation',diagnosis='POS',slide_id='slide'))
            with (base/'montage/montage_index.tsv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['slide_id','position','x','y','attention'],delimiter='\t');writer.writeheader()
                for i in range(16):writer.writerow(dict(slide_id='slide',position=i,x=448*i,y=0,attention=weights[i]))
            before = {str(p.relative_to(data)):p.read_bytes() for p in data.rglob('*') if p.is_file()}
            class FakeH5:
                attrs = dict(tile=224,tile_level0=224,target_mpp=.5)
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def __getitem__(self,key):
                    if key!='coords':raise AssertionError('Feature embeddings must not be loaded')
                    return coords
            class FakeSlide:
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def read_region(self,xy,level,size):return Image.new('RGB',size,(xy[0]//448,20,30))
            with patch.dict(sys.modules,{'h5py':SimpleNamespace(File=lambda *a:FakeH5()),
                                        'openslide':SimpleNamespace(OpenSlide=lambda *a:FakeSlide())}), \
                 patch('methylation_validate.validate_clients',return_value=dict(modality='multimodal',image_layout='tiles16',probes=list(range(849)))):
                pairs = prepare_pairs(data,base/'montage',base/'emb',base/'abmil',base/'svs',base/'output')
            self.assertEqual([x['image_count'] for x in pairs],[16,20])
            for a,b in zip(pairs[0]['images'],pairs[1]['images']):self.assertEqual(a.tobytes(),b.tobytes())
            self.assertEqual([im.getpixel((0,0))[0] for im in pairs[1]['images']],list(range(20)))
            after = {str(p.relative_to(data)):p.read_bytes() for p in data.rglob('*') if p.is_file()}
            self.assertEqual(before,after)
            self.assertFalse(json.loads((base/'output/tile_audit.json').read_text())[0]['extra_tiles_spacing_relaxed'])
            save_prepared_manifest(data, base/'output')
            (base/'reuse').mkdir()
            with patch('methylation_validate.validate_clients',return_value=dict(modality='multimodal',image_layout='tiles16',probes=list(range(849)))):
                reused = load_prepared_pairs(data, base/'output', base/'reuse')
                for a, b in zip(pairs, reused):
                    self.assertEqual(a['prompt'], b['prompt'])
                    self.assertEqual([im.tobytes() for im in a['images']], [im.tobytes() for im in b['images']])
                (base/'output/tiles/slide/tile-19.png').write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'tile checksum'):
                    load_prepared_pairs(data, base/'output', base/'reuse')


if __name__ == '__main__':
    unittest.main()
