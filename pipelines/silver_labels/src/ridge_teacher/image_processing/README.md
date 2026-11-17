# Image processing boundary

This package exports patient/sample/SVS linkage only. Actual image processing stays in the existing repository:

- data_preprocessing/wsi_embed.py: tissue/patch processing and encoder features.
- data_preprocessing/abmil_montage.py: supervised ABMIL and image preparation.
- data_preprocessing/tile_selection.py: selection rules.

For the new split, refit all supervised attention/tile selectors using allowed training patients only. Keep frozen pretrained encoders and label-independent raw patch caches reusable. Do not silently reuse a previous attention model trained on newly held-out patients.
