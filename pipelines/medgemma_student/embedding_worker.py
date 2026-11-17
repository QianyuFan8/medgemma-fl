"""One frozen encoder per process/GPU; atomic feature files and resumable receipts."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from .common import digest, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--worklist',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args()
    import torch
    from data_preprocessing.wsi_embed import load_encoder, embed_slide, DEFAULT_MODEL
    from .selector import load_feature
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rows=json.loads(Path(a.worklist).read_text());device=torch.device(a.device)
    args=SimpleNamespace(tile=224,target_mpp=.5,assume_mpp=None,coords_dir=None,sat_thresh=8,
        tissue_frac=.5,max_tiles=4000,seed=20261011,batch_size=16,num_workers=0,model=DEFAULT_MODEL,norm='bioptimus')
    model=transform=dim=None
    for row in rows:
        source=Path(row['slide_file']);target=out/(row['slide_id']+'.h5');receipt=target.with_suffix('.json')
        source_sha=digest(source)
        if target.exists() and receipt.exists():
            previous=json.loads(receipt.read_text())
            if previous['source_sha256']==source_sha and previous['feature_sha256']==digest(target):
                load_feature({'slide_id':row['slide_id'],'feature_file':str(target)})
                print('VERIFIED_FEATURE '+row['slide_id'],flush=True);continue
            raise ValueError('Existing feature receipt no longer matches; inspect before retrying')
        if target.exists():raise FileExistsError('Unreceipted feature file; inspect before retrying: '+str(target))
        if model is None:model,transform,dim=load_encoder(DEFAULT_MODEL,device,'bioptimus')
        partial=target.with_suffix('.h5.part')
        n,_,_=embed_slide(source,partial,model,transform,dim,args,device)
        load_feature({'slide_id':row['slide_id'],'feature_file':str(partial)})
        metadata=dict(source_sha256=source_sha,feature_sha256=digest(partial),tiles=n,model=DEFAULT_MODEL,
                      normalization='bioptimus',max_tiles=4000,sampling_seed=20261011,
                      sat_thresh=8,tissue_frac=.5)
        temp=receipt.with_suffix('.json.tmp');write_json(temp,metadata);os.replace(temp,receipt)
        os.replace(partial,target)
        print(f'EMBEDDED {row["slide_id"]} tiles={n}',flush=True)


if __name__=='__main__':main()
