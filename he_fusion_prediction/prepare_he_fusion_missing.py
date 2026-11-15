from pathlib import Path
import pandas as pd

root = Path('COMET_RMS_Data')
clinical = pd.read_excel(root/'COMET_RMS.xlsx', dtype=str).fillna('')
fusion = pd.read_excel(root/'COMET_RMS_fusion_status.xlsx', dtype=str).fillna('')
meta = pd.read_csv(root/'RMSs_meta.txt', sep='\t', dtype=str).fillna('')
for d in (clinical, fusion, meta):
    for c in d.columns: d[c] = d[c].astype(str).str.strip()
images = {p.stem for p in (root/'images').glob('*.svs')}
clinical['Slide ID'] = clinical['Slide Scan File'].str.replace(r'\.0$', '', regex=True)
df = clinical.merge(fusion[['Sample ID','Final Fusion Status']], on='Sample ID', how='left')
df = df.merge(meta[['Sentrix_ID','UID.Subject']], left_on='X850k Tumor File Name', right_on='Sentrix_ID', how='left')
df = df[df['Subtype Use For Clustering'].isin(['ARMS','ERMS']) & ~df['Final Fusion Status'].isin(['POS','NEG']) & df['Slide ID'].isin(images)].copy()
df = df.rename(columns={'Subtype Use For Clustering':'subtype','X850k Tumor File Name':'sentrix_id'})
df['patient_id'] = df['UID.Subject']; df['sample_id'] = df['Sample ID']
cols=['sample_id','patient_id','subtype','Slide ID','sentrix_id','Comp Bio ID','Site','Age At Collection (yr)','Sex']
df[cols].sort_values(['patient_id','sample_id']).to_csv('he_fusion_missing_manifest.csv', index=False)
print(f'Samples: {len(df)}; patients: {df.patient_id.nunique()}; unique slides: {df["Slide ID"].nunique()}')
print(df.groupby('subtype').size())
