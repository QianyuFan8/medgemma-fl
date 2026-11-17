import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipelines.medgemma_student.embeddings import embed_missing


class EmbeddingDispatch(unittest.TestCase):
    def test_gpu_shards_are_disjoint_and_resume_is_bound_to_configuration(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);rows=[]
            for i in range(5):
                slide=root/f'{i}.svs';slide.write_bytes(b'fixture')
                rows.append(dict(slide_id=str(i),slide_file=str(slide),feature_file=''))
            rows.append(dict(slide_id='cached',slide_file='',feature_file='already.h5'))
            plan=root/'plan.json';plan.write_text(json.dumps(dict(rows=rows,source_sha256={})))
            out=root/'out';seen=[]
            def fake_run(argv,**kwargs):
                worklist=Path(argv[argv.index('--worklist')+1])
                seen.append((kwargs['env']['CUDA_VISIBLE_DEVICES'],json.loads(worklist.read_text())))
                self.assertEqual(argv[argv.index('--device')+1],'cuda')
            with patch('pipelines.medgemma_student.embeddings.subprocess.run',side_effect=fake_run):
                embed_missing(plan,out,gpu_ids='0,1,2,3')
                self.assertEqual(sorted(x['slide_id'] for _,group in seen for x in group),['0','1','2','3','4'])
                self.assertEqual({gpu:len(group) for gpu,group in seen},{'0':2,'1':1,'2':1,'3':1})
                seen.clear();embed_missing(plan,out,gpu_ids='0,1,2,3',resume=True)
                self.assertEqual(len(seen),4)
                with self.assertRaisesRegex(ValueError,'differs'):
                    embed_missing(plan,out,gpu_ids='0',resume=True)
            self.assertEqual(json.loads((out/'completed.json').read_text())['max_tiles'],4000)

    def test_missing_wsi_and_duplicate_devices_fail_before_output(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);plan=root/'plan.json'
            plan.write_text(json.dumps(dict(rows=[dict(slide_id='1',slide_file='',feature_file='')],source_sha256={})))
            with self.assertRaisesRegex(ValueError,'distinct'):
                embed_missing(plan,root/'out',gpu_ids='0,0')
            with self.assertRaises(FileNotFoundError):embed_missing(plan,root/'out',gpu_ids='0')
            self.assertFalse((root/'out').exists())


if __name__=='__main__':unittest.main()
