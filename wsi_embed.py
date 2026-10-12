"""Tile H&E whole-slide images and embed every tissue tile with a pathology encoder.

Tissue detection follows CLAM's recipe (HSV saturation threshold on a thumbnail,
median blur). Tiles are 224 px at 0.5 um/px, the native setting of H-optimus-0/1.
If CLAM coordinate files are supplied (--coords-dir, patches/<slide>.h5 with a
`coords` dataset), they are used instead of the built-in segmentation.

Output: <output>/<slide>.h5 with `feats` [N, D] and `coords` [N, 2] (level-0 x, y)
plus attributes tile_level0, mpp0, target_mpp, tile, model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

DEFAULT_MODEL = "hf-hub:bioptimus/H-optimus-0"
# Bioptimus normalization constants (identical for H-optimus-0 and H-optimus-1).
NORM_MEAN = (0.707223, 0.578729, 0.703617)
NORM_STD = (0.211883, 0.230117, 0.177517)


def slide_mpp(slide, assume_mpp=None):
    for key in ("openslide.mpp-x", "aperio.MPP"):
        value = slide.properties.get(key)
        if value:
            return float(value)
    if assume_mpp:
        return float(assume_mpp)
    raise ValueError("Slide has no microns-per-pixel metadata; pass --assume-mpp only if the scanner setting is known")


def tissue_coords(slide, tile_level0, sat_thresh=8, tissue_frac=0.5, downsample=32):
    """Grid of level-0 tile origins whose thumbnail footprint is mostly tissue."""
    w, h = slide.dimensions
    thumb = slide.get_thumbnail((max(1, w // downsample), max(1, h // downsample))).convert("HSV")
    sat = np.asarray(thumb.split()[1].filter(ImageFilter.MedianFilter(7)))
    mask = sat > sat_thresh
    sy = h / mask.shape[0]
    sx = w / mask.shape[1]
    coords = []
    for y in range(0, h - tile_level0 + 1, tile_level0):
        r0, r1 = int(y / sy), max(int(y / sy) + 1, int((y + tile_level0) / sy))
        for x in range(0, w - tile_level0 + 1, tile_level0):
            c0, c1 = int(x / sx), max(int(x / sx) + 1, int((x + tile_level0) / sx))
            if mask[r0:r1, c0:c1].mean() >= tissue_frac:
                coords.append((x, y))
    return np.asarray(coords, dtype=np.int64).reshape(-1, 2)


def read_clam_coords(path):
    import h5py

    with h5py.File(path, "r") as f:
        coords = f["coords"][:]
        attrs = dict(f["coords"].attrs)
    return np.asarray(coords, dtype=np.int64), attrs


class TileDataset:
    """Reads tiles lazily; each DataLoader worker opens its own slide handle."""

    def __init__(self, slide_path, coords, tile_level0, tile, transform):
        self.slide_path, self.coords, self.tile_level0, self.tile, self.transform = slide_path, coords, tile_level0, tile, transform
        self._slide = None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        if self._slide is None:
            import openslide

            self._slide = openslide.OpenSlide(self.slide_path)
        x, y = (int(v) for v in self.coords[i])
        region = self._slide.read_region((x, y), 0, (self.tile_level0, self.tile_level0)).convert("RGB")
        if self.tile_level0 != self.tile:
            region = region.resize((self.tile, self.tile), Image.LANCZOS)
        return self.transform(region), i


def load_encoder(model_name, device):
    import timm
    import torch
    from torchvision import transforms

    model = timm.create_model(model_name, pretrained=True, init_values=1e-5, dynamic_img_size=False)
    model.eval().to(device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)])
    with torch.inference_mode():
        dim = model(torch.zeros(1, 3, 224, 224, device=device)).shape[-1]
    return model, transform, int(dim)


def embed_slide(slide_path, out_path, model, transform, dim, args, device):
    import h5py
    import openslide
    import torch

    slide = openslide.OpenSlide(str(slide_path))
    mpp0 = slide_mpp(slide, args.assume_mpp)
    tile_level0 = max(1, round(args.tile * args.target_mpp / mpp0))
    if args.coords_dir:
        coords, _ = read_clam_coords(Path(args.coords_dir) / f"{slide_path.stem}.h5")
    else:
        coords = tissue_coords(slide, tile_level0, args.sat_thresh, args.tissue_frac)
    slide.close()
    if args.max_tiles and len(coords) > args.max_tiles:
        rng = np.random.default_rng(args.seed)
        coords = coords[np.sort(rng.choice(len(coords), args.max_tiles, replace=False))]
    if len(coords) == 0:
        raise ValueError(f"No tissue tiles found in {slide_path}; inspect thresholds")
    dataset = TileDataset(str(slide_path), coords, tile_level0, args.tile, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    feats = np.zeros((len(coords), dim), dtype=np.float32)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for batch, idx in loader:
            feats[idx.numpy()] = model(batch.to(device)).float().cpu().numpy()
    with h5py.File(out_path, "w") as f:
        f.create_dataset("feats", data=feats)
        f.create_dataset("coords", data=coords)
        f.attrs.update(dict(tile_level0=tile_level0, mpp0=mpp0, target_mpp=args.target_mpp, tile=args.tile,
                            model=args.model, slide=str(slide_path.resolve())))
    return len(coords), tile_level0, mpp0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slide-dir", required=True)
    p.add_argument("--slides", nargs="*", help="Optional subset of slide stems; default = every .svs in --slide-dir")
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL, help="timm hf-hub id; use hf-hub:bioptimus/H-optimus-1 once approved")
    p.add_argument("--coords-dir", help="CLAM patches directory with <slide>.h5 coordinate files")
    p.add_argument("--tile", type=int, default=224)
    p.add_argument("--target-mpp", type=float, default=0.5)
    p.add_argument("--assume-mpp", type=float)
    p.add_argument("--sat-thresh", type=int, default=8)
    p.add_argument("--tissue-frac", type=float, default=0.5)
    p.add_argument("--max-tiles", type=int, default=0, help="Random cap on tiles per slide (0 = all)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20261011)
    args = p.parse_args()
    import torch

    device = torch.device(args.device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    slides = sorted(Path(args.slide_dir).glob("*.svs"))
    if args.slides:
        wanted = set(args.slides)
        slides = [s for s in slides if s.stem in wanted]
        missing = wanted - {s.stem for s in slides}
        if missing:
            raise FileNotFoundError(f"Slides not found: {sorted(missing)}")
    model, transform, dim = load_encoder(args.model, device)
    log = []
    for slide_path in slides:
        target = out / f"{slide_path.stem}.h5"
        if target.exists():
            print(f"skip {slide_path.stem}: exists")
            continue
        n, tile_level0, mpp0 = embed_slide(slide_path, target, model, transform, dim, args, device)
        log.append(dict(slide=slide_path.stem, tiles=n, tile_level0=tile_level0, mpp0=mpp0))
        print(f"{slide_path.stem}: {n} tiles, {tile_level0} px at level 0 (mpp {mpp0:.4f})")
    (out / "embedding_log.json").write_text(json.dumps(dict(model=args.model, dim=dim, tile=args.tile,
                                                            target_mpp=args.target_mpp, slides=log), indent=2) + "\n")


if __name__ == "__main__":
    main()
