"""Uniform scorer: iterate the registry, score every applicable cell x method at any n, and
AUTO-RUN THE GUARD (first-16 PSNR vs registry.PUBLISHED). A row whose guard fails prints `!! GUARD`
-- that is a wrong recipe, caught before you trust the full-n number.

    python -m experiments.score --n 200 [--metrics] [--cell gaussian_s05]
"""
import argparse, sys, numpy as np
sys.path.insert(0, ".")
from utils.metrics import psnr, ssim, lpips_metric
from . import registry as R
from . import adapters


def cell_dicts(only=None):
    out = []
    for nm, op, tag, sy in R.deblur_cells():
        out.append(dict(name=nm, op=op, tag=tag, sigma_y=sy, sr=False))
    for nm, kind, tag, sy in R.sr_cells():
        out.append(dict(name=nm, op=kind, tag=tag, sigma_y=sy, kind=kind, sr=True))
    return [c for c in out if (only is None or c["name"] == only)]


def score_cell(cell, n, dev, want_metrics):
    print(f"\n[{cell['name']}]  sigma_y={cell['sigma_y']}")
    for method, spec in R.METHODS.items():
        if cell["sr"] and spec.get("deblur_only"): continue
        if not cell["sr"] and spec.get("sr_only"): continue
        if not spec["applies"](cell["op"]): continue
        try:
            pairs, note = adapters.load_pairs(method, cell, n, dev)
        except Exception as e:
            print(f"  {method:12} !! could not load: {type(e).__name__}: {e}"); continue
        if not pairs:
            print(f"  {method:12} !! no recons"); continue
        P = np.array([psnr(r, c) for r, c in pairs])
        line = f"  {method:12} n={len(pairs):>3} PSNR {P.mean():6.2f}±{P.std():.2f}"
        if want_metrics:
            S = np.array([ssim(r, c) for r, c in pairs]); L = np.array([lpips_metric(r, c, dev) for r, c in pairs])
            line += f"  SSIM {S.mean():.3f}  LPIPS {L.mean():.3f}"
        # GUARD: first-16 PSNR vs published
        pub = R.PUBLISHED_PSNR.get((cell["name"], method))
        if pub is not None and len(pairs) >= 16:
            g16 = np.mean([psnr(*pairs[i]) for i in range(16)]); d = g16 - pub
            tag = "OK" if abs(d) <= R.GUARD_TOL else "!! GUARD"
            line += f"   [guard {g16:.2f} vs {pub:.2f}  {d:+.2f} {tag}]"
        if note: line += f"   ({note})"
        print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--metrics", action="store_true", help="also SSIM/LPIPS (slower)")
    ap.add_argument("--cell", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    for c in cell_dicts(args.cell):
        score_cell(c, args.n, args.device, args.metrics)


if __name__ == "__main__":
    main()
