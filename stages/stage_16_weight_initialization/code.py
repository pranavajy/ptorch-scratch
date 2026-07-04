"""Stage 15: Weight initialization.

Turns the variance-propagation rule Var(z) = n_in * Var(W) * Var(x) into the
Xavier/Glorot (tanh/linear) and He/Kaiming (relu) schemes plus a harness that
measures activation statistics across depth. Imports Dense (stage_11) and Tensor
(stage_12) as-is; builds samplers and a measurement harness on top.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

# Dense (stage_11) and Tensor (stage_12) via the shared dlfs shim, used as-is.
from dlfs import stage_import

Stage11_Dense = stage_import("stage_11", "Dense")
Stage12_Tensor = stage_import("stage_12", "Tensor")

Dense = Stage11_Dense
Tensor = Stage12_Tensor


def xavier_uniform(
    n_in: int, n_out: int, *, gain: float = 1.0, seed: Optional[int] = None
) -> np.ndarray:
    """Xavier/Glorot uniform init: U[-a, a], a = gain*sqrt(6/(n_in+n_out)). Returns (n_in, n_out)."""
    # TODO: sample U[-a, a] so Var(W) = gain**2 * 2 / (n_in + n_out).
    raise NotImplementedError("xavier_uniform")


def xavier_normal(
    n_in: int, n_out: int, *, gain: float = 1.0, seed: Optional[int] = None
) -> np.ndarray:
    """Xavier/Glorot normal init: N(0, std**2), std = gain*sqrt(2/(n_in+n_out)). Returns (n_in, n_out)."""
    # TODO: sample N(0, std**2) with std = gain*sqrt(2/(n_in+n_out)).
    raise NotImplementedError("xavier_normal")


def he_normal(n_in: int, n_out: int, *, seed: Optional[int] = None) -> np.ndarray:
    """He/Kaiming normal init for ReLU nets: N(0, 2/n_in). Returns (n_in, n_out)."""
    # TODO: sample N(0, 2/n_in) -- numerator 2 accounts for ReLU halving variance.
    raise NotImplementedError("he_normal")


def he_uniform(n_in: int, n_out: int, *, seed: Optional[int] = None) -> np.ndarray:
    """He/Kaiming uniform init for ReLU nets: U[-a, a], a = sqrt(6/n_in). Returns (n_in, n_out)."""
    # TODO: sample U[-a, a] so Var(W) = 2 / n_in.
    raise NotImplementedError("he_uniform")
