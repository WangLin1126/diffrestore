# Documents

| File | What |
|---|---|
| [methods_comparison.md](methods_comparison.md) | **DPS vs SMDC vs IHDM-HQS** — methods, models, results, and the full **derivation** of why SMDC = 1-step gradient and IHDM-HQS = full per-step MAP (with reverse-time notation + restoration procedure). |
| [MATH.md](MATH.md) | SMDC theory — scale-matched likelihood, existence of `Lₜ`, guidance modes, step-stability, assumptions. |
| [TASK.md](TASK.md) | Implementation spec / task plan (the cleaned form of the original design note). |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Running experiment log (gates, training, deblur results, DPS/HQS comparisons). |
| [ihdm_hqs_method.md](ihdm_hqs_method.md) | The heat_diffusion "scale-matched Bayesian posterior sampling" method note (source of IHDM-HQS). |
| [heat_diffusion_structure.md](heat_diffusion_structure.md) | Reference: structure of the original heat_diffusion repo (now vendored into `model/ihdm_backbone`, `model/dps_backbone`). |

*The original 1347-line design note (`task.md`) is captured in cleaned form across `MATH.md` + `TASK.md`;
the verbatim original can be restored from the build conversation on request.*
