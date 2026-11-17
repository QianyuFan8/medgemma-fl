import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from pipelines.medgemma_student.common import write_csv,write_json,digest
from pipelines.medgemma_student.dnanexus import select_files,required_names,download


class CloudTests(unittest.TestCase):
    def row(self):return dict(name='123.svs',id='file-abc',project='project-abc',size=4,state='closed')

    def test_duplicate_missing_and_archived_fail(self):
        r=self.row()
        self.assertEqual(select_files({'123.svs'},[r],'project-abc')[0]['id'],'file-abc')
        for rows in ([],[r,r],[{**r,'archivalState':'archived'}]):
            with self.assertRaises(ValueError):select_files({'123.svs'},rows,'project-abc')

    def test_download_receipt_resume_and_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);image=d/'images.csv';write_csv(image,[dict(patient_id='p1',slide_id='123')])
            manifest=d/'cloud.json';write_json(manifest,dict(files=[self.row()],total_bytes=4,image_manifest_sha256=digest(image)))
            dest=d/'wsi'
            def fake(cmd,**kwargs):Path(cmd[cmd.index('--output')+1]).write_bytes(b'data')
            with patch('pipelines.medgemma_student.dnanexus.subprocess.run',side_effect=fake) as run:
                download(manifest,image,dest,reserve_gib=0);download(manifest,image,dest,reserve_gib=0)
                self.assertEqual(run.call_count,1)
                self.assertIn('project-abc:file-abc',run.call_args.args[0])
                (dest/'123.svs').write_bytes(b'xxxx')
                with self.assertRaises(FileExistsError):download(manifest,image,dest,reserve_gib=0)

    def test_unsafe_slide_id_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'image.csv';write_csv(p,[dict(patient_id='p',slide_id='../123')])
            with self.assertRaises(ValueError):required_names(p)

    def test_wrong_cohort_cannot_download(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);image=d/'images.csv';write_csv(image,[dict(patient_id='p1',slide_id='123')])
            m=d/'cloud.json';write_json(m,dict(files=[self.row()],image_manifest_sha256='bad'))
            with patch('pipelines.medgemma_student.dnanexus.subprocess.run') as run:
                with self.assertRaises(ValueError):download(m,image,d/'out')
                run.assert_not_called()


if __name__=='__main__':unittest.main()
