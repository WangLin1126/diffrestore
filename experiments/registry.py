"""Single source of truth for the CelebA-HQ linear-restoration benchmark.

Everything that made the n=16 -> n=200 expansion painful is captured HERE, declaratively, in one
place: the test-set definition, every method's exact recipe (flags, per-cell hyper-parameters, the
sigma_y/2 noise convention), which cells a method applies to, how to load its recons for scoring
(the per-tool quirks), and the published first-16 guard values.

Expanding to a new n is then just:
    python -m experiments.run   --n 1000        # regenerate obs + run every applicable cell x method
    python -m experiments.score --n 1000        # uniform mean+/-std, with the first-16 guard auto-run

Nothing about a method's settings should live in a shell script or in your head. If a recipe is
wrong, the guard (first-16 must reproduce PUBLISHED below, because any n>=16 set is a superset of the
first 16) flags it BEFORE you trust the full-n numbers. See docs/PITFALLS.md for the war stories.
"""

# ---------------------------------------------------------------- test set (identical for all methods)
TESTSET = dict(
    source="korexyz/celeba-hq-256x256", split="validation", index="first-n",
    image_size=256, seed=0,
    # DEFINITIONAL INVARIANT: every method reconstructs the SAME images; obs is seeded so the first
    # 16 of any n-set byte-match the published n=16 set (the reproduction guard). Verify by md5.
)
ROOT = "results200"                       # cell outputs live here as <ROOT>/<cell>/<method>/recon
SHARED_CLEAN = f"{ROOT}/gaussian_s05/clean"   # the one 200-face clean set every method reads

# ---------------------------------------------------------------- cells
DEBLUR_OPS = ["gaussian", "motion", "defocus"]
DEBLUR_NOISE = {"s05": 0.05, "s10": 0.10, "s20": 0.20}     # sigma_y = std on [-1,1]
SR_KINDS = {"box": dict(aa_sigma=0.0), "aa": dict(aa_sigma=None)}   # None -> aa_sigma=scale=4
SR_NOISE = {"s01": 0.01, "s05": 0.05}

def deblur_cells():                          # ("gaussian_s05", op, tag, sigma_y)
    return [(f"{op}_{s}", op, s, sy) for op in DEBLUR_OPS for s, sy in DEBLUR_NOISE.items()]
def sr_cells():                              # ("sr_box_s01", kind, tag, sigma_y)
    return [(f"sr_{k}_{s}", k, s, sy) for k in SR_KINDS for s, sy in SR_NOISE.items()]

# ---------------------------------------------------------------- per-cell hyper-parameters
# IHDM frequency-aware reg gamma (`--freq_reg`). PER-CELL and PER-OPERATOR; never a single value.
IHDM_GAMMA_DEBLUR = {"gaussian": {"s05": 0.25, "s10": 0, "s20": 0},
                     "motion":   {"s05": 0.5,  "s10": 0, "s20": 0},
                     "defocus":  {"s05": 0,    "s10": 0, "s20": 0}}
IHDM_GAMMA_SR = {"box": {"s01": 0, "s05": 4}, "aa": {"s01": 0.5, "s05": 0}}

# TV: DIFFERENT script + params per operator (this bit us hard, see PITFALLS #3).
#   gaussian -> run_tv_hqs (DCT), NOISE-SCALED beta 2/3/6 ; motion -> run_tv_cg (reflect, else rings);
#   defocus  -> run_tv_hqs disk (beta 4). rho0=2 for hqs.
TV_RECIPE = {
    "gaussian": dict(script="run_tv_hqs", op="gaussian", extra="--blur_sigma 4 --rho0 2",
                     beta={"s05": 2, "s10": 3, "s20": 6}),
    "motion":   dict(script="run_tv_cg",  op=None,       extra="", beta={"s05": 4, "s10": 4, "s20": 4}),
    "defocus":  dict(script="run_tv_hqs", op="disk",     extra="--rho0 2", beta={"s05": 4, "s10": 4, "s20": 4}),
}

# Baseline noise convention: DPS/DDRM/DiffPIR/DDNM define noise on [0,1] then double to [-1,1] ->
# pass sigma_y/2. (DDRM ALSO doubles internally, so you still pass sigma_y/2.)
def half(sy): return round(sy / 2, 4)

# DDRM: only SEPARABLE operators. gaussian uses `deblur_aa` (our exact matched matrix G_s4, NOT the
# default near-uniform `deblur_gauss`!). SR box->sr4, aa->sr_aa. motion/defocus: N/A. timesteps=20.
DDRM_DEG = {"gaussian": "deblur_aa", "box": "sr4", "aa": "sr_aa"}

# DiffPIR blur mode / kernel per operator (matched kernels live in the scratchpad .npy files).
DIFFPIR_KERNEL = {"gaussian": "gauss_sigma4.npy", "motion": "results/motion_reflect/kernel.npy",
                  "defocus": "results200/defocus_s05/kernel.npy", "box": "box4.npy", "aa": "aa4.npy"}

# Fact worth pinning: our DCT-heat "sigma_blur = 4" is EXACTLY a spatial Gaussian of std 4 px
# (transfer = exp(-1/2 sigma^2 |xi|^2), separable), so a matched separable SVD/kernel reproduces it.

# ---------------------------------------------------------------- methods: applicability + scoring adapter
# adapter names are resolved in experiments/adapters.py. `applies(op)` gates non-universal methods.
METHODS = {
    "Observation": dict(adapter="observation", applies=lambda op: True),
    "TV":          dict(adapter="recon_dir:tv", applies=lambda op: True, deblur_only=True),
    "cold":        dict(adapter="recon_dir:cold", applies=lambda op: True, deblur_only=True),
    "IHDM":        dict(adapter="recon_dir:ihdm", applies=lambda op: True),  # SR: freqreg recon dir
    "DPS":         dict(adapter="dps", applies=lambda op: True, deblur_only=True, subset=100),  # <- n=100!
    "DDRM":        dict(adapter="ddrm", applies=lambda op: op in DDRM_DEG),   # not motion/defocus
    "DiffPIR":     dict(adapter="diffpir", applies=lambda op: True),
    "DDNM":        dict(adapter="ddnm", applies=lambda op: op == "box", sr_only=True),
}

# ---------------------------------------------------------------- published guard (first-16 PSNR)
# If a fresh run's first-16 PSNR deviates > TOL from these, the RECIPE is wrong -- fix before trusting.
GUARD_TOL = 0.4
PUBLISHED_PSNR = {
    # gaussian
    ("gaussian_s05","IHDM"):27.14,("gaussian_s10","IHDM"):26.47,("gaussian_s20","IHDM"):25.41,
    ("gaussian_s05","cold"):25.80,("gaussian_s10","cold"):25.11,("gaussian_s20","cold"):23.99,
    ("gaussian_s05","TV"):26.04,  ("gaussian_s10","TV"):25.23,  ("gaussian_s20","TV"):23.44,
    ("gaussian_s05","DPS"):23.84, ("gaussian_s10","DPS"):23.28, ("gaussian_s20","DPS"):22.57,
    ("gaussian_s05","DDRM"):26.78,("gaussian_s10","DDRM"):26.34,("gaussian_s20","DDRM"):25.66,
    ("gaussian_s05","DiffPIR"):26.21,("gaussian_s10","DiffPIR"):25.56,("gaussian_s20","DiffPIR"):24.72,
    # motion
    ("motion_s05","IHDM"):30.43,("motion_s10","IHDM"):28.41,("motion_s20","IHDM"):26.79,
    ("motion_s05","TV"):28.34,  ("motion_s10","TV"):26.64,  ("motion_s20","TV"):25.02,
    ("motion_s05","cold"):22.66,("motion_s10","cold"):22.05,("motion_s20","cold"):22.10,
    ("motion_s05","DPS"):24.90, ("motion_s10","DPS"):24.49, ("motion_s20","DPS"):23.74,
    ("motion_s05","DiffPIR"):28.87,("motion_s10","DiffPIR"):27.53,("motion_s20","DiffPIR"):26.25,
    # defocus
    ("defocus_s05","IHDM"):27.81,("defocus_s10","IHDM"):26.64,("defocus_s20","IHDM"):25.49,
    ("defocus_s05","TV"):26.16,  ("defocus_s10","TV"):25.16,  ("defocus_s20","TV"):23.91,
    ("defocus_s05","cold"):24.10,("defocus_s10","cold"):24.50,("defocus_s20","cold"):24.00,
    ("defocus_s05","DPS"):23.23, ("defocus_s10","DPS"):22.98, ("defocus_s20","DPS"):22.46,
    ("defocus_s05","DiffPIR"):26.88,("defocus_s10","DiffPIR"):25.94,("defocus_s20","DiffPIR"):24.80,
    # SR
    ("sr_box_s01","IHDM"):28.60,("sr_box_s05","IHDM"):27.80,("sr_aa_s01","IHDM"):27.19,("sr_aa_s05","IHDM"):25.40,
    ("sr_box_s01","DDRM"):27.84,("sr_box_s05","DDRM"):27.31,("sr_aa_s01","DDRM"):26.66,("sr_aa_s05","DDRM"):25.50,
    ("sr_box_s01","DDNM"):28.38,("sr_box_s05","DDNM"):27.78,
    ("sr_box_s01","DiffPIR"):27.43,("sr_box_s05","DiffPIR"):26.81,("sr_aa_s01","DiffPIR"):26.14,("sr_aa_s05","DiffPIR"):24.61,
}
