"""Fit combined ABMIL on all labeled patients and score unlabeled H&E samples."""
from pathlib import Path
import argparse, json, random
import h5py, numpy as np, pandas as pd, torch

from train_patient_abmil_combined import GatedABMIL, load_bag

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--labeled-manifest',required=True); p.add_argument('--missing-manifest',required=True)
    p.add_argument('--hoptimus',required=True); p.add_argument('--embeddinggemma',required=True)
    p.add_argument('--output',required=True); p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--lr',type=float,default=2e-4); p.add_argument('--device',default='cuda'); p.add_argument('--seed',type=int,default=20261114)
    a=p.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device=torch.device(a.device); out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    lab=pd.read_csv(a.labeled_manifest); miss=pd.read_csv(a.missing_manifest)
    lab['y']=lab['label'].map({'NEG':0,'POS':1})
    hdir,edir=Path(a.hoptimus),Path(a.embeddinggemma)
    all_sids=sorted(set(lab['Slide ID'].astype(str))|set(miss['Slide ID'].astype(str)))
    bags={sid:load_bag(sid,hdir,edir) for sid in all_sids}
    model=GatedABMIL().to(device); n=len(lab); pos=int(lab.y.sum())
    loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor([n/(2*(n-pos)),n/(2*pos)],device=device))
    opt=torch.optim.Adam(model.parameters(),lr=a.lr,weight_decay=1e-5)
    for epoch in range(a.epochs):
        model.train()
        for _,r in lab.sample(frac=1,random_state=a.seed+epoch).iterrows():
            logits,_=model(torch.from_numpy(bags[str(r['Slide ID'])]).to(device))
            loss=loss_fn(logits[None,:],torch.tensor([int(r.y)],device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    torch.save({'state_dict':model.state_dict(),'dim':2304,'classes':['NEG','POS'],'epochs':a.epochs},out/'model.pt')
    model.eval(); rows=[]
    with torch.no_grad():
        for _,r in miss.iterrows():
            sid=str(r['Slide ID']); prob=float(model(torch.from_numpy(bags[sid]).to(device))[0].softmax(-1)[1])
            rows.append({**r.to_dict(),'predicted_pos_probability':prob,'predicted_fusion_status':'POS' if prob>=0.5 else 'NEG'})
    pd.DataFrame(rows).to_csv(out/'missing_fusion_predictions.csv',index=False)
    print(json.dumps({'labeled_samples':len(lab),'labeled_patients':lab.patient_id.nunique(),'missing_samples':len(miss),'output':str(out)},indent=2))

if __name__=='__main__': main()
