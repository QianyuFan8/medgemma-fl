"""Refit gated ABMIL and extract strict spatial16; never fit/evaluate holdouts."""
import json
from pathlib import Path
import random
import numpy as np
from .common import digest, load_plan, write_json
from data_preprocessing.abmil_montage import build_model


def strict_spatial16(weights, coords, tile_size):
    weights, coords = np.asarray(weights), np.asarray(coords)
    if weights.ndim!=1 or coords.shape!=(len(weights),2) or tile_size<=0:
        raise ValueError('Invalid attention/coordinates')
    if not np.isfinite(weights).all() or not np.isfinite(coords).all() or len(np.unique(coords,axis=0))!=len(coords):
        raise ValueError('Nonfinite or duplicate coordinates')
    selected=[]
    # Scan all cached candidate tiles in attention order. No distance relaxation.
    for i in np.argsort(-weights,kind='stable'):
        if not selected or np.linalg.norm(coords[selected]-coords[i],axis=1).min() >= 2*tile_size:
            selected.append(int(i))
        if len(selected)==16:
            return selected
    raise ValueError('Cannot select 16 tiles with >=2 tile-width origin distance; manual review required')


def load_feature(row):
    import h5py
    path=Path(row['feature_file'])
    if not row['feature_file'] or not path.is_file():
        raise FileNotFoundError(f"Missing H-Optimus feature: {row['slide_id']}")
    with h5py.File(path,'r') as f:
        feats,coords,attrs=f['feats'][:].astype(np.float32),f['coords'][:],dict(f.attrs)
    model=str(attrs.get('model',''))
    if 'H-optimus-1' not in model or feats.ndim!=2 or feats.shape[1]!=1536:
        raise ValueError(f'Expected frozen H-Optimus-1 1536-D features: {path}')
    if len(feats)<16 or coords.shape!=(len(feats),2) or not np.isfinite(feats).all():
        raise ValueError(f'Invalid feature bag: {path}')
    if int(attrs['tile'])!=224 or abs(float(attrs['target_mpp'])-.5)>1e-6:
        raise ValueError('Expected 224px tiles at 0.5 um/px')
    return feats,coords,attrs


def fit_selector(plan_path, output, device='cpu', epochs=50, lr=1e-4, seed=20261117, max_tiles=4096):
    import torch
    plan=load_plan(plan_path);output=Path(output)
    if output.exists():raise FileExistsError(output)
    if epochs<1 or lr<=0 or max_tiles<16:raise ValueError('Invalid selector settings')
    # Validate complete feature coverage before spending time fitting.
    missing=[r['slide_id'] for r in plan['rows'] if not r['feature_file'] or not Path(r['feature_file']).is_file()]
    if missing:raise FileNotFoundError(f'Missing features for {len(missing)} slides; see asset inventory')
    torch.manual_seed(seed);np.random.seed(seed);rng=random.Random(seed)
    train=[r for r in plan['rows'] if r['split']=='train']
    model=build_model(1536).to(device);opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5)
    bags=[]
    for row in train:
        feats,_,_=load_feature(row)
        if len(feats)>max_tiles:
            feats=feats[np.sort(np.random.default_rng(seed).choice(len(feats),max_tiles,replace=False))]
        bags.append((torch.from_numpy(feats),int(row['label_name']=='POS'),row['student_loss_weight']))
    if {b[1] for b in bags}!={0,1}:raise ValueError('Selector training requires both labels')
    history=[]
    for epoch in range(1,epochs+1):
        model.train();order=list(range(len(bags)));rng.shuffle(order);total=0.
        for i in order:
            feats,label,weight=bags[i]
            logits,_=model(feats.to(device))
            loss=torch.nn.functional.cross_entropy(logits[None],torch.tensor([label],device=device))*weight
            opt.zero_grad();loss.backward();opt.step();total+=loss.item()
        history.append(dict(epoch=epoch,weighted_train_loss=total/len(bags)))
        print(json.dumps(history[-1]),flush=True)
    output.mkdir(parents=True);(output/'attention').mkdir()
    torch.save(dict(state_dict=model.cpu().state_dict(),dim=1536,hidden=512,attn=256,dropout=.25),output/'model.pt')
    model.to(device).eval();feature_hashes={}
    # Do not access holdout labels here: export attention only, no scores/metrics.
    with torch.inference_mode():
        for row in plan['rows']:
            feats,_,_=load_feature(row)
            _,attention=model(torch.from_numpy(feats).to(device))
            np.save(output/'attention'/f"{row['slide_id']}.npy",attention.cpu().numpy())
            feature_hashes[row['slide_id']]=digest(row['feature_file'])
    write_json(output/'selector.json',dict(plan_sha256=digest(plan_path),epochs=epochs,lr=lr,seed=seed,
        max_training_tiles=max_tiles,train_patients=[r['patient_id'] for r in train],
        label_weights={'gold':1.,'silver':.3},history=history,feature_sha256=feature_hashes,
        attention_sha256={r['slide_id']:digest(output/'attention'/f"{r['slide_id']}.npy") for r in plan['rows']},
        model_sha256=digest(output/'model.pt'),holdout_labels_used=False,checkpoint_selection='fixed final epoch'))


def extract_tiles(plan_path, selector, output):
    import openslide
    from PIL import Image
    plan=load_plan(plan_path);selector=Path(selector);output=Path(output)
    if output.exists():raise FileExistsError(output)
    receipt=json.loads((selector/'selector.json').read_text())
    if receipt['plan_sha256']!=digest(plan_path) or receipt['holdout_labels_used']:
        raise ValueError('Selector provenance differs from current split')
    if receipt['model_sha256']!=digest(selector/'model.pt'):
        raise ValueError('Selector model changed')
    expected={r['patient_id'] for r in plan['rows'] if r['split']=='train'}
    if set(receipt['train_patients'])!=expected:raise ValueError('Selector training cohort differs')
    # Full preflight before any image output; use exact scanner coordinates.
    selected=[]
    for row in plan['rows']:
        if not row['slide_file'] or not Path(row['slide_file']).is_file():
            raise FileNotFoundError(f"Missing WSI: {row['slide_id']}")
        if digest(row['feature_file'])!=receipt['feature_sha256'][row['slide_id']]:
            raise ValueError('Feature bag changed since ABMIL fitting')
        a=selector/'attention'/f"{row['slide_id']}.npy"
        if digest(a)!=receipt['attention_sha256'][row['slide_id']]:raise ValueError('Attention changed')
        _,coords,attrs=load_feature(row);weights=np.load(a)
        indices=strict_spatial16(weights,coords,int(attrs['tile_level0']))
        selected.append((row,indices,coords,attrs,weights))
    output.mkdir(parents=True);audit=[]
    for row,indices,coords,attrs,weights in selected:
        size=int(attrs['tile_level0']);canvas=Image.new('RGB',(896,896))
        with openslide.OpenSlide(row['slide_file']) as slide:
            if 'dimensions' in attrs and tuple(map(int,attrs['dimensions']))!=slide.dimensions:
                raise ValueError('WSI dimensions disagree with embedding source')
            from data_preprocessing.wsi_embed import slide_mpp
            if abs(slide_mpp(slide)-float(attrs['mpp0']))>1e-5:
                raise ValueError('WSI resolution disagrees with embedding source')
            for j,i in enumerate(indices):
                x,y=map(int,coords[i])
                if x<0 or y<0 or x+size>slide.dimensions[0] or y+size>slide.dimensions[1]:
                    raise ValueError('Selected tile outside WSI')
                tile=slide.read_region((x,y),0,(size,size)).convert('RGB').resize((224,224),Image.Resampling.LANCZOS)
                canvas.paste(tile,((j%4)*224,(j//4)*224))
        name=f"{row['slide_id']}.png";canvas.save(output/name)
        audit.append(dict(patient_id=row['patient_id'],slide_id=row['slide_id'],file=name,
            image_sha256=digest(output/name),indices=indices,coords=coords[indices].tolist(),
            attention=weights[indices].tolist(),tile_level0=size,source_wsi_sha256=digest(row['slide_file'])))
    write_json(output/'tiles.json',dict(plan_sha256=digest(plan_path),selector_sha256=digest(selector/'selector.json'),
        distance_rule='Euclidean distance between tile origins >= 2 * tile_level0; no relaxation',
        storage='896x896 lossless container; loader crops into 16 independent 224x224 images',rows=audit))
