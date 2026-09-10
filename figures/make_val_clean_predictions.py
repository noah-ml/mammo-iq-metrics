#!/usr/bin/env python3
"""
make_val_clean_predictions.py
Produce CLEAN (undegraded) predictions for the VALIDATION split, so calibration
parameters (temperature T, Platt a/b) can be fit on validation and evaluated on the
held-out test set.

Run this on the cluster (where the checkpoint and the 1024x384 tar shards live), then
copy the output CSV back next to the test results. It reuses the EXACT model build,
checkpoint loading, preprocessing and AMP policy from evaluate_degradations.py, so the
val logits are produced identically to the test logits in run 957382.

Usage
-----
    python make_val_clean_predictions.py \
        --checkpoint /path/to/ckpt_epoch006_auc0.8103.pth \
        --shards-dir /path/to/vindr_tar_shards_1024x384_45positive \
        --splits-csv /path/to/master_splits_1024x384_5positive.csv \
        --arch convnext_tiny \
        --output val_clean_predictions.csv

Output columns: image_id, label_45, logit, prob   (clean condition only, val split)
"""
from __future__ import annotations
import argparse, io, json, math, sys, tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# reuse the verified pieces from the evaluation script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from evaluate_degradations import build_model, load_checkpoint, preprocess  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--shards-dir", required=True)
    p.add_argument("--splits-csv", required=True)
    p.add_argument("--arch", default="convnext_tiny",
                   choices=["convnext_tiny", "resnet18", "efficientnet_b4"])
    p.add_argument("--output", default="val_clean_predictions.csv")
    p.add_argument("--split", default="val", help="split name in the shard manifest (val)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model = build_model(args.arch, dropout=0.3)
    load_checkpoint(args.checkpoint, model)
    model.eval(); model = model.float().to(device)

    splits = pd.read_csv(args.splits_csv); splits["image_id"] = splits["image_id"].astype(str)
    splits["label_45"] = (splits["breast_birads"] >= 4).astype(int)
    lab = splits.set_index("image_id")["label_45"]

    manifest = pd.read_csv(Path(args.shards_dir) / "metadata" / "shard_manifest.csv")
    man = manifest[manifest["split"] == args.split].reset_index(drop=True)
    print(f"{args.split} images: {len(man)}")

    tar_cache: dict[str, tarfile.TarFile] = {}
    def get_tar(path):
        if path not in tar_cache:
            tar_cache[path] = tarfile.open(path, "r:*")
        return tar_cache[path]

    rows = []
    for _, r in tqdm(man.iterrows(), total=len(man), desc=f"{args.split} clean"):
        iid = str(r["image_id"])
        shard = str(Path(args.shards_dir) / args.split / str(r["shard"]))
        try:
            tf = get_tar(shard)
            img = np.load(io.BytesIO(tf.extractfile(f"{iid}.npy").read())).astype(np.float32)
        except Exception as exc:
            print(f"skip {iid}: {exc}"); continue
        t = preprocess(img).float().to(device)
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logit = model(t).view(-1)
            else:
                logit = model(t).view(-1)
            prob = float(torch.sigmoid(logit).float().cpu().item())
        rows.append(dict(image_id=iid, label_45=int(lab.get(iid, 0)),
                         logit=float(logit.float().cpu().item()), prob=prob))

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {args.output}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
