"""Per-step run logger (acceptance criterion 9 in TASK.md)."""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict


@dataclass
class StepLog:
    step: int
    t: float
    t_next: float
    residual_norm: float
    correction_norm: float
    prior_step_norm: float
    state_min: float
    state_max: float
    state_consistency: float = float("nan")


@dataclass
class RunLogger:
    verbose: bool = True
    steps: List[StepLog] = field(default_factory=list)

    def log(self, s: StepLog) -> None:
        self.steps.append(s)
        if self.verbose:
            print(
                f"[{s.step:03d}] t={s.t:>7.3f}->{s.t_next:<7.3f} "
                f"|r|={s.residual_norm:.4e} |corr|={s.correction_norm:.4e} "
                f"|prior|={s.prior_step_norm:.4e} range=[{s.state_min:+.3f},{s.state_max:+.3f}] "
                f"sc={s.state_consistency:.3e}"
            )

    def to_dicts(self) -> List[Dict]:
        return [asdict(s) for s in self.steps]

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dicts(), f, indent=2)
