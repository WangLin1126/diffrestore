"""TV and DPS baselines for SMDC super-resolution -- head-to-head on the SAME operator and the
SAME observations as scripts/sr_demo.py (which saves out/clean + out/observation, the physical LR).

Both baselines invert the identical forward A = ops.superres.SuperResolution (antialias Gaussian +
avg-pool decimation), so the ONLY difference from SMDC+IHDM is the prior:
  * TV  -- HQS with a Chambolle TV prox + spatial CG data step (same solver skeleton as
           scripts/run_tv_cg.py, operator swapped to SR). No learned prior.
  * DPS -- hot diffusion (vendored model/dps_backbone + ffhq_10m) with x0-guidance through the SAME
           SuperResolution operator (not DPS's own bicubic Resizer), so the degradation matches.

  python scripts/sr_baselines.py --scale 4 --methods tv dps --in_dir results/sr_demo \
     --out results/sr_demo
"""
import os, sys, glob, types, argparse
from pathlib import Path
from functools import partial
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image

from ops.superres import SuperResolution
from ops.motion_spatial import cg_solve
from utils.metrics import psnr, ssim, lpips_metric


def load_png(path, size, dev):
    im = Image.open(path).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    x = torch.from_numpy(np.asarray(im, np.float32)).permute(2, 0, 1) / 127.5 - 1.0
    return x[None].to(dev)


def report(name, S, out_png):
    m = {k: float(np.mean(v)) for k, v in S.items()}
    print(f"[SR {name}]  PSNR {m['out']:.2f} dB  SSIM {m['ssim']:.3f}  LPIPS {m['lpips']:.3f}  "
          f"-> {out_png}", flush=True)
    return m


# --------------------------------------------------------------------------- TV + CG
def run_tv(cleans, obs_lr, A, H, dev, args, out_dir):
    from skimage.restoration import denoise_tv_chambolle
    rec_dir = os.path.join(out_dir, "tv_recon"); os.makedirs(rec_dir, exist_ok=True)

    def tvprox(x, w):
        z = denoise_tv_chambolle(x[0].detach().cpu().permute(1, 2, 0).numpy(), weight=w, channel_axis=-1)
        return torch.from_numpy(z).permute(2, 0, 1)[None].float().to(dev)

    S = {"out": [], "ssim": [], "lpips": []}
    for i, (x0, y) in enumerate(zip(cleans, obs_lr)):
        sig2 = (args.sigma_y * y.abs().mean()) ** 2                      # matches sr_demo's relative noise
        x = F.interpolate(y, size=(H, H), mode="bicubic", align_corners=False)
        rho = args.rho0; Aty = A.adjoint(y)
        for _ in range(args.T):
            z = tvprox(x, args.beta / rho)
            def M(v, rho=rho): return (1.0 / sig2) * A.adjoint(A.forward(v)) + rho * v
            x = cg_solve(M, (1.0 / sig2) * Aty + rho * z, z, iters=args.cg_iters)
            rho *= args.gamma
        x = x.clamp(-1, 1)
        save_image((x[0] + 1) / 2, os.path.join(rec_dir, f"{i:05d}.png"))
        S["out"].append(psnr(x, x0)); S["ssim"].append(ssim(x, x0)); S["lpips"].append(lpips_metric(x, x0, dev))
        print(f"  TV  img {i}: {S['out'][-1]:.2f} dB  SSIM {S['ssim'][-1]:.3f}", flush=True)
    return report(f"TV+CG beta={args.beta}", S, os.path.join(out_dir, "tv_recon"))


# --------------------------------------------------------------------------- DPS
def run_dps(cleans, obs_lr, scale, dev, args, out_dir):
    import yaml
    REPO = Path(__file__).resolve().parent.parent
    DPS_ROOT = REPO / "model" / "dps_backbone"
    sys.path.insert(0, str(DPS_ROOT))
    mb = types.ModuleType("motionblur"); mbm = types.ModuleType("motionblur.motionblur")
    mbm.Kernel = type("K", (), {}); mb.motionblur = mbm
    sys.modules.setdefault("motionblur", mb); sys.modules.setdefault("motionblur.motionblur", mbm)
    from guided_diffusion.condition_methods import get_conditioning_method
    from guided_diffusion.gaussian_diffusion import create_sampler
    from guided_diffusion.unet import create_model

    mc = yaml.safe_load(open(str(DPS_ROOT / "configs/model_config.yaml")))
    mc["model_path"] = str(DPS_ROOT / "models" / "ffhq_10m.pt")
    H = int(mc["image_size"])
    model = create_model(**mc).to(dev).eval()

    A = SuperResolution(scale, H, channels=3, device=dev, dtype=torch.float32)   # SAME operator

    class Operator:                                                              # DPS forward/transpose API
        __name__ = "sr"
        def forward(self, data, **kw): return A.forward(data)
        def transpose(self, data, **kw): return A.adjoint(data)
    class GaussianNoiser: __name__ = "gaussian"

    cond = get_conditioning_method("ps", Operator(), GaussianNoiser(), scale=args.dps_scale)
    sampler = create_sampler(**yaml.safe_load(open(str(DPS_ROOT / "configs/diffusion_config.yaml"))))
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=cond.conditioning)

    rec_dir = os.path.join(out_dir, "dps_recon"); os.makedirs(rec_dir, exist_ok=True)
    S = {"out": [], "ssim": [], "lpips": []}
    for i, (x0, y) in enumerate(zip(cleans, obs_lr)):
        x_start = torch.randn(1, 3, H, H, device=dev).requires_grad_()
        rec = sample_fn(x_start=x_start, measurement=y, record=False, save_root=str(rec_dir))
        rec = rec.detach().clamp(-1, 1)
        save_image((rec[0] + 1) / 2, os.path.join(rec_dir, f"{i:05d}.png"))
        S["out"].append(psnr(rec, x0)); S["ssim"].append(ssim(rec, x0)); S["lpips"].append(lpips_metric(rec, x0, dev))
        print(f"  DPS img {i}: {S['out'][-1]:.2f} dB  SSIM {S['ssim'][-1]:.3f}", flush=True)
    return report(f"DPS scale={args.dps_scale}", S, os.path.join(out_dir, "dps_recon"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--methods", nargs="+", default=["tv", "dps"], choices=["tv", "dps"])
    ap.add_argument("--in_dir", default="results/sr_demo", help="has clean/ and observation/ from sr_demo")
    ap.add_argument("--out", default="results/sr_demo")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--sigma_y", type=float, default=0.01)
    ap.add_argument("--device", default="cuda:0")
    # TV knobs
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--rho0", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=1.2)
    ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--cg_iters", type=int, default=12)
    # DPS knobs
    ap.add_argument("--dps_scale", type=float, default=0.3)
    args = ap.parse_args()
    dev, H, s = args.device, args.image_size, args.scale

    cpaths = sorted(glob.glob(os.path.join(args.in_dir, "clean", "*.png")))[:args.n]
    opaths = sorted(glob.glob(os.path.join(args.in_dir, "observation", "*.png")))[:args.n]
    assert cpaths and len(cpaths) == len(opaths), f"need matching clean/observation in {args.in_dir}"
    cleans = [load_png(p, H, dev) for p in cpaths]
    obs_lr = [load_png(p, H // s, dev) for p in opaths]                # physical LR at H/s
    print(f"SR baselines: n={len(cleans)} scale={s} obs={tuple(obs_lr[0].shape[-2:])} "
          f"methods={args.methods}", flush=True)

    A = SuperResolution(s, H, channels=3, device=dev, dtype=torch.float32)
    if "tv" in args.methods:
        run_tv(cleans, obs_lr, A, H, dev, args, args.out)
    if "dps" in args.methods:
        run_dps(cleans, obs_lr, s, dev, args, args.out)


if __name__ == "__main__":
    main()
