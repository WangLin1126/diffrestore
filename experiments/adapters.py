"""Recon-loading adapters: one per method, each encoding that tool's scoring quirk in ONE place.
Every adapter returns a list of (recon_tensor, reference_tensor) pairs in [-1,1], so the scorer is
uniform. This is where DDRM's reordered `orig`, DiffPIR's process-panel + lambda sweep, DDNM's `Apy`
subdir, and DPS's n=100 subset are handled -- never re-derive them ad hoc again.
"""
import os, glob, numpy as np, torch
from PIL import Image
import torch.nn.functional as F
from utils.pipeline import load_png, normalize_operator
from ops.superres import SuperResolution
from . import registry as R

DIFFPIR_RES = "model/DiffPIR_backbone/results"
DDRM_SAMP = "model/ddrm_backbone/exp/image_samples"
DDNM_SAMP = "model/ddnm_backbone/exp/image_samples"
BM = {"gaussian": "gauss", "motion": "motion", "defocus": "defocus"}

def _pngs(d, n): return sorted(glob.glob(f"{d}/*.png"))[:n]
def _clean(n, dev): return [load_png(p, 256, dev) for p in _pngs(R.SHARED_CLEAN, n)]
def _last_block(path, dev):
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)   # SR panel HxW*k -> last 256
    return torch.from_numpy(im[:, -256:, :]).permute(2, 0, 1)[None].to(dev) / 127.5 - 1

def load_pairs(method, cell, n, dev):
    """cell = {name, op, tag, sigma_y, kind?}. Returns (pairs, note)."""
    a = R.METHODS[method]["adapter"]; name, op, tag, sy = cell["name"], cell["op"], cell["tag"], cell["sigma_y"]
    C = _clean(n, dev)

    if a == "observation":
        if name.startswith("sr_"):                                   # bicubic-upsampled LR
            A_raw = SuperResolution(4, 256, channels=3, aa_sigma=R.SR_KINDS[cell["kind"]]["aa_sigma"],
                                    decimation="avgpool", device=dev, dtype=torch.float32)
            A, lam = normalize_operator(A_raw, 256, dev, torch.float32); obs = []
            for i in range(n):
                torch.manual_seed(i); y = A.forward(C[i]); y = y + (sy / lam) * torch.randn_like(y)
                obs.append(F.interpolate(y * lam, size=(256, 256), mode="bicubic", align_corners=False))
            return list(zip(obs, C)), "bicubic"
        rec = [load_png(p, 256, dev) for p in _pngs(f"{R.ROOT}/{name}/observation", n)]
        return list(zip(rec, C)), ""

    if a.startswith("recon_dir:"):
        m = a.split(":")[1]
        if m == "ihdm" and name.startswith("sr_"):
            g = R.IHDM_GAMMA_SR[cell["kind"]][tag]
            d = f"{R.ROOT}/{name}/sy{sy:g}_reg{g:g}/recon"
        else:
            d = f"{R.ROOT}/{name}/{m}/recon"
        rec = [load_png(p, 256, dev) for p in _pngs(d, n)]
        return list(zip(rec, C)), ""

    if a == "dps":                                                    # first-100 subset
        k = min(R.METHODS["DPS"]["subset"], n)
        rec = [load_png(p, 256, dev) for p in _pngs(f"{R.ROOT}/{name}/dps/heat_blur/recon", k)]
        return list(zip(rec, C[:len(rec)])), f"n={len(rec)} subset"

    if a == "ddrm":                                                   # {id}_-1.png vs its own orig_{id}
        deg_key = op if op in ("gaussian",) else cell["kind"]
        dd = f"{DDRM_SAMP}/{_ddrm_name(name, op, cell.get('kind'))}"
        rec = [load_png(f"{dd}/{i}_-1.png", 256, dev) for i in range(n)]
        ref = [load_png(f"{dd}/orig_{i}.png", 256, dev) for i in range(n)]   # DDRM REORDERS -> use its orig
        return list(zip(rec, ref)), "vs own orig"

    if a == "ddnm":                                                   # {i}_*.png vs Apy/orig_{i}
        d = f"{DDNM_SAMP}/{_ddnm_name(name)}"; rec = []; ref = []
        for i in range(n):
            fs = [x for x in glob.glob(f"{d}/{i}_*.png") if "/Apy/" not in x]
            rec.append(load_png(fs[0], 256, dev)); ref.append(load_png(f"{d}/Apy/orig_{i}.png", 256, dev))
        return list(zip(rec, ref)), "vs Apy/orig"

    if a == "diffpir":
        if name.startswith("sr_"):                                   # panel last-block, best-PSNR lambda
            from utils.metrics import psnr
            sig = f"{R.half(sy):g}"; d = glob.glob(glob.escape(DIFFPIR_RES) + f"/our{n}_sr_*sigma{sig}*{cell['kind']}")[0]
            lams = sorted({os.path.basename(f).split("lambda_")[1].split("_")[0]
                           for f in glob.glob(glob.escape(d) + "/*.png") if "lambda_" in f}, key=float)
            best = None
            for lam in lams:
                pairs = []
                for i in range(n):
                    fs = glob.glob(glob.escape(d) + f"/{i:05d}_*lambda_{lam}_*.png")
                    if not fs: pairs = None; break
                    pairs.append((_last_block(fs[0], dev), C[i]))
                if pairs is None: continue
                mp = np.mean([psnr(r, c) for r, c in pairs])
                if best is None or mp > best[1]: best = (pairs, mp, lam)
            return best[0], f"last-block, best lambda={best[2]}"
        sig = f"{R.half(sy):g}"; d = glob.glob(glob.escape(DIFFPIR_RES) + f"/our{n}_deblur_*sigma{sig}_*blurmode{BM[op]}")[0]
        rec = [load_png(p, 256, dev) for p in sorted(glob.glob(glob.escape(d) + "/*_diffusion_ffhq_10m.png"))[:n]]
        return list(zip(rec, C)), ""
    raise ValueError(f"unknown adapter {a}")

# name maps for the external sample dirs (as produced by experiments/run.py)
def _ddrm_name(name, op, kind):
    if op == "gaussian": return {"gaussian_s05":"ddrm_gaa05","gaussian_s10":"ddrm_gaa10","gaussian_s20":"ddrm_gaa20"}[name]
    return {"sr_box_s01":"ddrm_box01","sr_box_s05":"ddrm_box05","sr_aa_s01":"ddrm_aa01","sr_aa_s05":"ddrm_aa05"}[name]
def _ddnm_name(name): return {"sr_box_s01":"ddnm_box01","sr_box_s05":"ddnm_box05"}[name]
