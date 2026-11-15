import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

df = pd.read_csv('he_fusion_spatial16_combined_manifest.csv')
patient = df.groupby('patient_id', as_index=False).agg(label=('label','first'))
if df.groupby('patient_id')['label'].nunique().max() != 1:
    raise ValueError('Conflicting labels within a patient')
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261114)
patient['fold'] = -1
for fold, (_, test) in enumerate(cv.split(patient, patient['label'], patient['patient_id'])):
    patient.loc[test, 'fold'] = fold
out = df.merge(patient, on='patient_id', how='left', suffixes=('', '_patient'))
out = out.drop(columns=['label_patient'])
out.to_csv('he_fusion_spatial16_combined_manifest_with_folds.csv', index=False)
print(out.groupby(['fold','label'])['patient_id'].nunique())
print(f'Samples: {len(out)}; patients: {out.patient_id.nunique()}')
