"""Read-only audit of a legacy ERMS/ARMS panel against canonical metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd


def inspect(prepared, metadata, output):
    root=Path(prepared);out=Path(output)
    if root.resolve()==out.resolve():raise ValueError('Do not write audit outputs over source data')
    spec=json.loads((root/'features.json').read_text())
    if set(spec.get('classes', [])) != {'ERMS', 'ARMS'}:
        raise ValueError('This inspection report is specifically for legacy ERMS/ARMS preparations')
    summary=json.loads((root/'selection_summary.json').read_text())
    splits=pd.read_csv(root/'splits.tsv',sep='\t',dtype=str,keep_default_na=False)
    meta=pd.read_csv(metadata,sep='\t',dtype=str,keep_default_na=False)
    for c in meta:meta[c]=meta[c].str.strip()
    key='Sentrix_ID' if 'Sentrix_ID' in meta else 'Sample_Name'
    if meta[key].duplicated().any():raise ValueError('Duplicate methylation metadata keys')
    splits['sentrix']=splits['beta_column'].str.removesuffix('.Ave_Beta')
    linked=splits.merge(meta[[key,'UID.Subject']],left_on='sentrix',right_on=key,how='left',validate='many_to_one')
    known=linked.dropna(subset=['UID.Subject'])
    crossing=known.groupby('UID.Subject')['split'].nunique()
    crossing=set(crossing[crossing>1].index)
    matrices={};checks={};all_ids=[]
    for split in ['train','validation','test']:
        df=pd.read_csv(root/f'{split}.csv');all_ids.extend(df.patient_id)
        b=df[spec['probes']]
        matrices[split]={'rows':len(df),'columns':len(df.columns),'labels':df.diagnosis.value_counts().to_dict()}
        ref=splits[splits.split.eq(split)]
        checks[split]={'feature_order_matches':df.columns.tolist()==['patient_id','diagnosis']+spec['probes'],
            'patient_set_matches_manifest':set(df.patient_id)==set(ref.patient_id),
            'no_missing_beta':bool(b.notna().all().all()),'beta_in_unit_interval':bool(((b>=0)&(b<=1)).all().all())}
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file() and p.name!='.DS_Store'}
    report={'source':str(root.resolve()),'classes':spec['classes'],'selected_cpgs':len(spec['probes']),
        'selection_summary':summary,'matrices':matrices,'checks':checks,'original_id_rows':len(splits),
        'unique_original_ids':len(set(all_ids)),'canonical_patients':known['UID.Subject'].nunique(),
        'unresolved_rows':int(linked['UID.Subject'].isna().sum()),'canonical_patients_crossing_splits':len(crossing),
        'sha256':hashes,'usable_as_new_fusion_panel':False,
        'reason':'ERMS/ARMS supervised selection and noncanonical splits; reselect from full beta using NEG/POS and canonical patient folds.'}
    out.mkdir(parents=True,exist_ok=False)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    linked.to_csv(out/'canonical_linkage.tsv',sep='\t',index=False)
    linked[linked['UID.Subject'].isin(crossing)].to_csv(out/'cross_split_patients.tsv',sep='\t',index=False)
    (out/'README.md').write_text(f'''# Data 文件夹检查

原文件保持不变。这里是旧 ERMS/ARMS 任务的完整预处理输出，不是 fusion panel。

- 参与训练筛选的候选 CpG：{summary['tested_nonconstant_cpgs']:,}。
- 选中 CpG：{len(spec['probes'])}（top 0.1%）；实际 FDR <0.05 的入选数：{summary['selected_bh_fdr_below_005']}。
- train/validation/test：{matrices['train']['rows']}/{matrices['validation']['rows']}/{matrices['test']['rows']} 行。
- 原 ID 下 {len(splits)} 条记录；metadata UID.Subject 下 {report['canonical_patients']} 位病人。
- {len(crossing)} 位病人在多个 split 出现；详见 cross_split_patients.tsv。
- 文件中标记没有可用本地 H&E，不代表最新 DNAnexus 也没有；必须重新核对新切片清单。

| 文件 | 内容 |
|---|---|
| train.csv / validation.csv / test.csv | patient_id、旧任务标签及选中的 849 个 CpG beta 值 |
| features.tsv / features.json / selected_cpgs.tsv | CpG 顺序、训练中位数、P 值/FDR、panel 标识 |
| limma_training_only.tsv | 全部被检验 CpG 的统计结果，不包含完整原始 beta 矩阵 |
| eligible_cpgs.tsv | 候选 CpG 名称列表，不是它们的 beta 值 |
| splits.tsv / sample_audit.tsv | 被纳入记录、病人/样本/切片信息、划分和 QC 状态 |
| preprocessing.rds | CpG 名单、中位数、配置和 classes；不是已训练分类器 |
| config.json / selection_summary.json / cohort_provenance.json / FEATURES.md / R_sessionInfo.txt | 筛选参数、来源、统计及运行环境 |

这些文件不含原始全 CpG beta、SVS、H-optimus embeddings、MedGemma/ABMIL 模型权重。
CpG 顺序、beta 范围、manifest 病人集合检查见 report.json；完整文件 SHA256 也已记录。
不能把 ERMS/ARMS 改成 NEG/POS 后将此目录用于正式 fusion 评估。
''')
    print(json.dumps({k:v for k,v in report.items() if k not in ['sha256','selection_summary']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True);p.add_argument('--metadata',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();inspect(a.prepared,a.metadata,a.output)
