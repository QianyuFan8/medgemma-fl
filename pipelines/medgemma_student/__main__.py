import argparse
import json


def main():
    p=argparse.ArgumentParser(description='Prepare gold/silver MedGemma inputs without reusing old supervised selectors.')
    sub=p.add_subparsers(dest='command',required=True)
    x=sub.add_parser('dx-inventory');x.add_argument('--image-manifest',required=True);x.add_argument('--output',required=True)
    x=sub.add_parser('dx-download');x.add_argument('--cloud-manifest',required=True);x.add_argument('--image-manifest',required=True)
    x.add_argument('--output',required=True);x.add_argument('--reserve-gib',type=float,default=75.)
    x=sub.add_parser('plan');x.add_argument('--returned',required=True);x.add_argument('--cohort',required=True)
    x.add_argument('--output',required=True);x.add_argument('--feature-dir',action='append',default=[]);x.add_argument('--slide-dir',action='append',default=[])
    x=sub.add_parser('embed-missing');x.add_argument('--plan',required=True);x.add_argument('--output',required=True)
    x.add_argument('--device',default='cpu')
    x.add_argument('--gpu-ids',help='Independent frozen encoder workers, e.g. 0,1,2,3; not MedGemma DDP')
    x.add_argument('--resume',action='store_true')
    x=sub.add_parser('fit-selector');x.add_argument('--plan',required=True);x.add_argument('--output',required=True)
    x.add_argument('--device',default='cpu');x.add_argument('--epochs',type=int,default=50);x.add_argument('--lr',type=float,default=1e-4)
    x.add_argument('--seed',type=int,default=20261117);x.add_argument('--max-tiles',type=int,default=4096)
    x=sub.add_parser('extract');x.add_argument('--plan',required=True);x.add_argument('--selector',required=True);x.add_argument('--output',required=True)
    x=sub.add_parser('export');x.add_argument('--plan',required=True);x.add_argument('--tiles',required=True);x.add_argument('--output',required=True)
    x=sub.add_parser('verify');x.add_argument('--data',required=True)
    a=p.parse_args()
    if a.command=='dx-inventory':
        from .dnanexus import inventory
        inventory(a.image_manifest,a.output)
    elif a.command=='dx-download':
        from .dnanexus import download
        download(a.cloud_manifest,a.image_manifest,a.output,a.reserve_gib)
    elif a.command=='plan':
        from .plan import prepare_plan
        report=prepare_plan(a.returned,a.cohort,a.output,a.feature_dir,a.slide_dir)
        print(json.dumps({k:report[k] for k in ('counts','missing_features','missing_wsi')},indent=2))
    elif a.command=='embed-missing':
        from .embeddings import embed_missing
        embed_missing(a.plan,a.output,a.device,a.gpu_ids,a.resume)
    elif a.command=='fit-selector':
        from .selector import fit_selector
        fit_selector(a.plan,a.output,a.device,a.epochs,a.lr,a.seed,a.max_tiles)
    elif a.command=='extract':
        from .selector import extract_tiles
        extract_tiles(a.plan,a.selector,a.output)
    elif a.command=='export':
        from .inputs import export_inputs
        print(json.dumps(export_inputs(a.plan,a.tiles,a.output)['counts'],indent=2))
    else:
        from .inputs import verify_export
        print(json.dumps(verify_export(a.data)['counts'],indent=2))


if __name__=='__main__':main()
