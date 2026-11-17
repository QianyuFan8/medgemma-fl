"""Build the explicit local research-data handoff; never include credentials or WSI.

Run only when intentionally preparing a bundle for an authorized research VM.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    repo=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capstone-root',type=Path,default=repo.parent)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();base=a.capstone_root.resolve();out=a.output.resolve()
    archive=out/'medgemma_student_vm_20261117_v2.tar.gz'
    if archive.exists():raise FileExistsError('Use a new output directory: '+str(archive))
    files={}
    def add(path,name):
        if not path.is_file():raise FileNotFoundError(path)
        if name in files:raise ValueError('Duplicate bundle path: '+name)
        files[name]=path
    for name in ('README.md','LICENSE','pyproject.toml','requirements.txt','requirements-vm.txt','requirements-wsi.txt'):
        add(repo/name,'repo/'+name)
    for dirname in ('src','train','inference','experiments','data_preprocessing','scripts','tests','pipelines','methylation'):
        for file in sorted((repo/dirname).rglob('*')):
            rel=file.relative_to(repo)
            if (file.is_file() and not any(part.startswith('.') or part=='__pycache__' for part in rel.parts)
                and 'site-packages' not in rel.parts and file.suffix in {'.py','.sh','.R','.md','.toml','.yml','.txt'}):
                add(file,'repo/'+rel.as_posix())
    for name in ('REPOSITORY_GUIDE_ZH.md','TRAINING_PERFORMANCE.md','entrypoint_migration.json'):
        add(repo/'docs'/name,'repo/docs/'+name)
    for name in ('student_train.csv','silver_labels.csv','image_input_manifest.csv','paired_selected_cpg_beta_imputed.csv','panel_provenance.json'):
        add(base/'return_results/silver'/name,'inputs/returned/silver/'+name)
    for name in ('teacher_selection.json','limma/teacher/selected_cpgs.txt'):
        add(base/'return_results'/name,'inputs/returned/'+name)
    add(base/'outputs/silver_labels_20261117/cohort_v4/patients.csv','inputs/cohort/patients.csv')
    add(base/'outputs/fusion_vm_849_20261027/README.md','provenance/archived_feature_commands_20261027.md')
    add(base/'outputs/medgemma_student_v4_20261117/STATUS.json','provenance/preparation_status.json')
    add(repo/'pipelines/medgemma_student/VM_DNANEXUS.md','START_HERE.md')
    plan=json.loads((base/'outputs/medgemma_student_v4_20261117/plan/plan.json').read_text())
    features=[r for r in plan['rows'] if r['feature_file']]
    if len(features)!=48 or len(plan['rows'])!=131:raise ValueError('Frozen cohort/cache count changed')
    for row in features:add(Path(row['feature_file']),'features/hoptimus1/'+row['slide_id']+'.h5')
    manifest={'schema':2,'code_layout':'installed medgemma_fl package with train/inference entrypoints',
              'prepared_date':'2026-11-17','cached_feature_count':48,'missing_feature_count':83,
              'required_wsi_count':131,'gpu_tested':False,
              'sha256':{name:digest(file) for name,file in sorted(files.items())}}
    payload=(json.dumps(manifest,indent=2)+'\n').encode();out.mkdir(parents=True,exist_ok=True)
    partial=archive.with_suffix('.gz.part')
    with tarfile.open(partial,'w:gz',compresslevel=1) as tar:
        for name,file in sorted(files.items()):tar.add(file,arcname='medgemma_student_vm/'+name,recursive=False)
        info=tarfile.TarInfo('medgemma_student_vm/bundle_manifest.json');info.size=len(payload)
        tar.addfile(info,io.BytesIO(payload))
    with tarfile.open(partial,'r:gz') as tar:
        verified=set()
        for member in tar:
            rel=member.name.removeprefix('medgemma_student_vm/')
            if rel=='bundle_manifest.json':continue
            if not member.isfile() or rel not in manifest['sha256']:raise ValueError('Unexpected archive member')
            h=hashlib.sha256();f=tar.extractfile(member)
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            if h.hexdigest()!=manifest['sha256'][rel]:raise ValueError('Archive changed: '+rel)
            verified.add(rel)
        if verified!=set(files):raise ValueError('Archive coverage changed')
    partial.rename(archive)
    (out/'bundle_manifest.json').write_bytes(payload)
    (out/'START_HERE.md').write_text((repo/'pipelines/medgemma_student/VM_DNANEXUS.md').read_text())
    (out/(archive.name+'.sha256')).write_text(digest(archive)+'  '+archive.name+'\n')
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'verified_files':len(files)},indent=2))


if __name__=='__main__':main()
