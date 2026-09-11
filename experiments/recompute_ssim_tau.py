"""
Recompute masked SSIM + supervisor-reference tau for the deployed 957382 results.

- Reuses degradations.get_breast_mask / apply_degradation and
  evaluate_degradations._setup_per_image so the degraded images are byte-identical
  to the deployed run (noise seed = img_idx = image order in the deployed CSV).
- masked SSIM  : mean of the skimage SSIM map over the breast mask.
- reference tau: fast port of the supervisor's av_thresh_noise (local-extrema NMS,
                 adaptive threshold grid), restricted to the breast-mask interior.
- Validation mode (--limit N): also recomputes UNMASKED ssim and checks it matches
  the deployed ssim column (proves pipeline+seed fidelity), and checks fast-tau
  against the supervisor's av_thresh_noise.

Usage:
  python recompute_ssim_tau.py --limit 12            # validation subset
  python recompute_ssim_tau.py                       # full 4000 (background)
"""
import argparse, io, os, sys, tarfile, math
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import degradations as deg
from evaluate_degradations import _setup_per_image, DEGRADATION_CONDITIONS

from paths import DATA_DIR, OUT_DIR  # noqa: E402

#: Evaluation output to recompute against, as written by
#: evaluate_degradations.py (unmasked SSIM, in-script tau).
DEPLOYED = os.environ.get(
    "MAMMO_DEPLOYED_CSV",
    str(DATA_DIR / "convnext_nojitter_degradation_results_blurfix_cnr_jp2_tau_STRICT_CLEAN.csv"))
#: Test-split tar shards holding the preprocessed 1024x384 arrays.
TEST_TARS = os.environ.get("MAMMO_TEST_TARS",
                           str(DATA_DIR / "vindr_tar_shards_1024x384_45positive" / "test"))
OUT = str(OUT_DIR / "recomputed_ssim_tau.csv")

# ---------------- fast reference tau (supervisor algorithm) ----------------
def _bexc(x, i_nn=0):
    B_left=np.roll(x,1,1); B_right=np.roll(x,-1,1); B_up=np.roll(x,1,0); B_down=np.roll(x,-1,0)
    parts=[x-B_left, x-B_right, x-B_up, x-B_down]
    if i_nn==0:
        parts += [x-np.roll(B_left,1,0), x-np.roll(B_left,-1,0), x-np.roll(B_right,1,0), x-np.roll(B_right,-1,0)]
    img_n=np.stack(parts,axis=-1)
    img_n[0,:,:]=0; img_n[-1,:,:]=0; img_n[:,0,:]=0; img_n[:,-1,:]=0
    min_pos=np.clip(img_n.min(axis=-1),0,None)
    max_neg=np.clip(img_n.max(axis=-1),None,0)
    return min_pos+np.abs(max_neg)   # nonzero only at local extrema

def _tau_from_bexc(pos, nthreshmax=1000):
    if pos.size==0: return 0.0
    min_d=float(pos.min()); max_d=float(pos.max())
    if max_d<=0: return 0.0
    if min_d==0:
        nthresh=nthreshmax; delta=max_d/(nthreshmax-1)
    else:
        nthresh=int(np.ceil(max_d/min_d))+1; delta=min_d
        if nthresh>nthreshmax: nthresh=nthreshmax; delta=max_d/(nthreshmax-1)
    thr=delta*np.arange(nthresh)
    ps=np.sort(pos)
    n_t=ps.size-np.searchsorted(ps,thr,side="right")
    denom=float(n_t.sum())
    if denom==0: return 0.0
    return float(np.sum(thr*n_t)/denom)

def ref_tau(img, mask=None, i_nn=0, nthreshmax=1000):
    x=img.astype(np.float64)
    b=_bexc(x,i_nn)
    if mask is not None:
        m=mask.astype(bool)
        interior=(m[1:-1,1:-1]&m[:-2,:-2]&m[:-2,1:-1]&m[:-2,2:]&m[1:-1,:-2]&m[1:-1,2:]&m[2:,:-2]&m[2:,1:-1]&m[2:,2:])
        bc=b[1:-1,1:-1][interior]
    else:
        bc=b.ravel()
    return _tau_from_bexc(bc[bc>0], nthreshmax)

def contrast_iqr(img, mask):
    """Noise-robust global contrast: interquartile range (p75-p25) of breast-mask
    intensities. Scales by alpha under I' = c + alpha*(I-c); ignores the noise-driven
    tails, so far less noise-inflated than the std-based RMS contrast."""
    v = img[mask.astype(bool)]
    if v.size < 4:
        return float("nan")
    q75, q25 = np.percentile(v.astype(np.float64), [75, 25])
    return float(q75 - q25)

def masked_ssim(clean, deg_img, mask):
    _, smap = structural_similarity(clean.astype(np.float32), deg_img.astype(np.float32),
                                    data_range=1.0, full=True)
    m=mask.astype(bool)
    return float(smap[m].mean()) if m.any() else float(smap.mean())

# ---------------- validation of fast tau vs supervisor ----------------
def validate_fast_tau():
    try:
        from tau_ref_numpy import av_thresh_noise
    except Exception as e:
        print("  (supervisor tau_ref_numpy not importable:", e, ")"); return
    rng=np.random.default_rng(1)
    print("  fast ref_tau (no mask) vs supervisor av_thresh_noise:")
    for s in [0.01,0.03,0.08]:
        img=np.clip(0.5+rng.normal(0,s,(96,96)),0,1)
        a=ref_tau(img, mask=None)
        b=float(av_thresh_noise(img.astype(np.float64), nthreshmax=1000, i_nn=0)[0][0])
        print(f"    sigma={s:.2f}  fast={a:.6f}  supervisor={b:.6f}  |diff|={abs(a-b):.2e}")

# ---------------- tar index ----------------
def build_index():
    idx={}
    for fn in sorted(os.listdir(TEST_TARS)):
        if not fn.endswith(".tar"): continue
        p=os.path.join(TEST_TARS,fn)
        with tarfile.open(p) as tf:
            for m in tf.getmembers():
                if m.name.endswith(".npy"):
                    idx[m.name[:-4]]=(p, m.name)
    return idx

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    args=ap.parse_args()

    print("[1] validating fast reference tau ...")
    validate_fast_tau()

    print("[2] reading deployed CSV order ...")
    df=pd.read_csv(DEPLOYED, low_memory=False)
    order=list(dict.fromkeys(df["image_id"].tolist()))   # img_idx = appearance order
    dep_ssim={(r.image_id,r.degradation_type,int(r.severity)):r.ssim
              for r in df.itertuples(index=False)}
    print(f"    {len(order)} images, {len(df)} rows")

    print("[3] indexing test tars ...")
    tar_idx=build_index()
    print(f"    {len(tar_idx)} npy in test shards; coverage of CSV images: "
          f"{sum(im in tar_idx for im in order)}/{len(order)}")

    images = order if args.limit==0 else order[:args.limit]
    # --- resume: preload already-computed images and skip them (img_idx preserved via enumerate) ---
    recs=[]; val_absdiff=[]; done_ids=set()
    if not args.limit and os.path.exists(OUT):
        prev=pd.read_csv(OUT)
        # only keep fully-complete images (all 31 conditions) as done
        counts=prev.groupby("image_id").size()
        complete=set(counts[counts==len(DEGRADATION_CONDITIONS)].index)
        prev=prev[prev["image_id"].isin(complete)]
        recs=prev[["image_id","degradation_type","severity","ssim_masked","tau_ref","contrast_iqr"]].values.tolist()
        done_ids=complete
        print(f"    RESUME: {len(done_ids)} images already done ({len(recs)} rows), skipping them")
    for img_idx, image_id in enumerate(order):
        if args.limit and img_idx>=args.limit: break
        if image_id in done_ids:
            continue
        if image_id not in tar_idx:
            continue
        tp,mem=tar_idx[image_id]
        with tarfile.open(tp) as tf:
            img_np=np.load(io.BytesIO(tf.extractfile(mem).read())).astype(np.float32)
        try:
            mask=deg.get_breast_mask(img_np)
            sigma0,patch_bbox=_setup_per_image(img_np, mask)
        except Exception as e:
            print("  setup fail",image_id,e); continue
        for deg_type,severity in DEGRADATION_CONDITIONS:
            if deg_type=="clean":
                img_deg=img_np
            else:
                img_deg=deg.apply_degradation(img_np, deg_type, severity, mask=mask,
                                              sigma0=sigma0, pixel_spacing_mm=0.07, seed=img_idx)
            sm=masked_ssim(img_np, img_deg, mask)
            tr=ref_tau(img_deg, mask=mask)
            cir=contrast_iqr(img_deg, mask)
            rec=[image_id, deg_type, severity, round(sm,6), round(tr,6), round(cir,6)]
            if args.limit:  # validation: also unmasked ssim vs deployed
                un=float(structural_similarity(img_np.astype(np.float32), img_deg.astype(np.float32), data_range=1.0))
                dep=dep_ssim.get((image_id,deg_type,severity), np.nan)
                if not (isinstance(dep,float) and math.isnan(dep)):
                    val_absdiff.append(abs(un-float(dep)))
                rec += [round(un,6), dep]
            recs.append(rec)
        if not args.limit and (img_idx % 200 == 0):
            pd.DataFrame(recs, columns=["image_id","degradation_type","severity","ssim_masked","tau_ref","contrast_iqr"]).to_csv(OUT, index=False)
            print(f"  ...{img_idx}/{len(order)} images ({len(recs)} rows) checkpointed", flush=True)
        if args.limit:
            print(f"  [{img_idx}] {image_id} done")

    cols=["image_id","degradation_type","severity","ssim_masked","tau_ref","contrast_iqr"]
    if args.limit: cols += ["ssim_unmasked_recomputed","ssim_deployed"]
    out=pd.DataFrame(recs, columns=cols)
    outpath = OUT if not args.limit else os.path.join(REPO,"recompute_validation.csv")
    out.to_csv(outpath, index=False)
    print(f"[4] wrote {outpath}  ({len(out)} rows)")
    if args.limit and val_absdiff:
        va=np.array(val_absdiff)
        print(f"\n  VALIDATION unmasked-SSIM vs deployed:  max|diff|={va.max():.2e}  mean|diff|={va.mean():.2e}  n={va.size}")
        print("  (if max|diff| ~<1e-3 the degradation pipeline + noise seed are faithful)")

if __name__=="__main__":
    main()
