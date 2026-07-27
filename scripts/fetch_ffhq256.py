"""Download + extract FFHQ-256 (zip shards) into a single uint8 cache tensor.

Produces data/ffhq256/ffhq256_uint8.pt  (N,3,256,256) for fast DataParallel training.
"""
import os
import sys
import io
import time
import zipfile
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

REPO = "pravsels/FFHQ_256"
OUT = "data/ffhq256"


def main():
    n_shards = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    os.makedirs(OUT, exist_ok=True)
    imgs = []
    t0 = time.time()
    for i in range(1, n_shards + 1):
        fn = f"shard_{i}_of_14.zip"
        print(f"[{i}/{n_shards}] downloading {fn} ...", flush=True)
        p = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=fn,
                            local_dir=os.path.join(OUT, "zips"))
        with zipfile.ZipFile(p) as z:
            names = [n for n in z.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg"))]
            for n in names:
                im = Image.open(io.BytesIO(z.read(n))).convert("RGB")
                if im.size != (256, 256):
                    im = im.resize((256, 256), Image.LANCZOS)
                imgs.append(np.asarray(im, dtype=np.uint8))
        print(f"    total images so far: {len(imgs)}  ({time.time()-t0:.0f}s)", flush=True)
    arr = np.stack(imgs)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    out_path = os.path.join(OUT, "ffhq256_uint8.pt")
    torch.save(t, out_path)
    print(f"saved {t.shape} -> {out_path}  ({t.numel()/1e9:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
