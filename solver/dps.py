"""DPS solver — spare comparison path.

DPS's data consistency is fundamentally different from the SMDC/HQS correctors in this
skeleton: it defines the likelihood on a Tweedie clean estimate x0_hat(x_t) and pushes
`grad_{x_t} ||y - A x0_hat||^2` by backpropagating through the denoiser (whereas SMDC/HQS
define the likelihood directly on the current-scale state x_t and never backprop through the
prior). It also only runs on a hot VP diffusion prior. It is therefore not a drop-in Corrector;
the full DPS restoration lives in `scripts/run_dps.py` (via the vendored guided-diffusion repo).

See `docs/methods_comparison.md` for the derivation contrasting all three.
"""
RUNNER = "scripts/run_dps.py"
