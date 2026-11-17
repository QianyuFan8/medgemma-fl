"""Freeze exact DNAnexus file IDs, then resume verified WSI downloads via dx CLI.

Credentials remain in dx's normal login environment; never placed in manifests.
"""
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from .common import digest, indexed, read_csv, write_json

PROJECT='project-JBXfX9Q0vQQYJ9k4BxGVyYzJ'
FOLDER='/COMET_RMS'


def dx_json(args):
    result=subprocess.run(['dx',*args,'--json'],check=True,capture_output=True,text=True)
    return json.loads(result.stdout)


def required_names(image_manifest):
    rows=read_csv(image_manifest);indexed(rows);indexed(rows,'slide_id')
    if any(not re.fullmatch(r'[0-9]+',r['slide_id']) for r in rows):
        raise ValueError('Unexpected slide ID: refusing an unsafe or ambiguous filename')
    return {r['slide_id']+'.svs' for r in rows}


def select_files(names, found, project):
    by_name=defaultdict(list)
    for entry in found:
        d=entry.get('describe',entry)
        if d.get('name') in names:by_name[d['name']].append((entry,d))
    selected=[]
    for name in sorted(names):
        matches=by_name[name]
        if len(matches)!=1:raise ValueError(f'{name}: expected exactly one cloud file, found {len(matches)}')
        entry,d=matches[0];fid=entry.get('id',d.get('id'))
        if not fid or not fid.startswith('file-') or d.get('state')!='closed' or d.get('size',0)<=0:
            raise ValueError(f'{name}: file not closed or metadata incomplete')
        if d.get('archivalState','live')!='live':raise ValueError(f'{name}: cloud file is archived')
        if entry.get('project',d.get('project',project))!=project:raise ValueError('Wrong cloud project')
        selected.append(dict(name=name,id=fid,project=project,size=d['size'],
                             created=d.get('created'),modified=d.get('modified')))
    return selected


def inventory(image_manifest, output, project=PROJECT, folder=FOLDER):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    names=required_names(image_manifest)
    # Read-only CLI calls; no project selection mutation, upload, or job launch.
    project_info=dx_json(['describe',project])
    found=dx_json(['find','data','--path',project+':'+folder,'--norecurse',
                   '--class','file','--name','*.svs','--state','closed','--verbose'])
    files=select_files(names,found,project)
    report=dict(schema=1,project=project,folder=folder,region=project_info.get('region'),
        observed_utc=datetime.now(timezone.utc).isoformat(),image_manifest_sha256=digest(image_manifest),
        files=files,total_bytes=sum(r['size'] for r in files))
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output,report)
    print(json.dumps(dict(files=len(files),GiB=report['total_bytes']/2**30,region=report['region']),indent=2))
    return report


def verified(path,row,receipts):
    prior=receipts.get(row['name'])
    return (path.is_file() and path.stat().st_size==row['size'] and prior is not None
            and prior['id']==row['id'] and prior.get('project')==row['project']
            and prior['sha256']==digest(path))


def download(cloud_manifest,image_manifest,destination,reserve_gib=75.):
    manifest=json.loads(Path(cloud_manifest).read_text());destination=Path(destination)
    if manifest['image_manifest_sha256']!=digest(image_manifest):raise ValueError('Cloud inventory belongs to another patient cohort')
    if {r['name'] for r in manifest['files']}!=required_names(image_manifest):raise ValueError('Cloud file coverage differs')
    if len({r['name'] for r in manifest['files']})!=len(manifest['files']):raise ValueError('Duplicate download targets')
    if reserve_gib<0:raise ValueError('Reserve must be nonnegative')
    destination.mkdir(parents=True,exist_ok=True)
    # Prevent two launchers from writing the same .part or receipt concurrently.
    import fcntl
    with (destination/'.download.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt_path=destination/'download_receipts.json'
        receipts=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        todo=[r for r in manifest['files'] if not verified(destination/r['name'],r,receipts)]
        required=sum(r['size'] for r in todo)+int(reserve_gib*2**30)
        if shutil.disk_usage(destination).free<required:
            raise OSError(f'Insufficient free space: need at least {required/2**30:.1f} GiB (includes reserve)')
        for i,row in enumerate(manifest['files'],1):
            target=destination/row['name']
            if row not in todo:
                print(f'[{i}/{len(manifest["files"])}] verified; skip {row["name"]}',flush=True);continue
            # Never overwrite an existing final filename with unverifiable content.
            if target.exists():raise FileExistsError(f'Unverified existing file: {target}; inspect/move it before retrying')
            partial=target.with_suffix('.svs.part')
            print(f'[{i}/{len(manifest["files"])}] dx download {row["name"]}',flush=True)
            # dx --overwrite resumes matching partial signatures; normal validation
            # stays enabled (never --lightweight). Download by immutable ID, not glob.
            subprocess.run(['dx','download',row['project']+':'+row['id'],
                            '--output',str(partial),'--overwrite','--no-progress'],check=True)
            if partial.stat().st_size!=row['size']:raise ValueError('Downloaded byte count differs')
            sha=digest(partial)
            receipts[row['name']]={'id':row['id'],'project':row['project'],'size':row['size'],'sha256':sha}
            # Write receipt first. If interrupted before rename, next run safely
            # re-verifies/resumes the partial through dx instead of trusting it.
            temp=receipt_path.with_suffix('.tmp');write_json(temp,receipts);os.replace(temp,receipt_path)
            os.replace(partial,target)
        write_json(destination/'download_complete.json',dict(cloud_manifest_sha256=digest(cloud_manifest),
            image_manifest_sha256=digest(image_manifest),files=len(manifest['files']),
            total_bytes=manifest['total_bytes'],download_receipts_sha256=digest(receipt_path)))
