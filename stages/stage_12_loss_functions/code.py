"""Stage 12: Loss functions for a SINGLE example (mse, mae, cross-entropy) + sum/mean.

First of two loss stages. Everything here is for ONE example -- no batch axis -- so each
loss is a single scalar. stage_13 generalizes the classification losses to a batch of
(B, C) logits; doing one example first separates the softmax/CE idea from batching.

Two parts, both on top of the autodiff ``Tensor``:
  1. ADD the ``sum`` / ``mean`` reductions (with backward) -- earlier stages shipped
     none, and every loss needs them to collapse an array to a scalar.
  2. ADD the losses ``mse_loss`` / ``mae_loss`` / ``cross_entropy_loss`` (with a stable
     ``softmax`` / ``log_softmax``).

Build losses from Tensor ops so gradients flow through ``backward()``; NumPy is for
forward array construction only, never grads.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

# Reuse the autodiff Tensor engine.  We extend the broadcast-capable Tensor from
# stage_11 (it adds the broadcasting-correct backward); THIS stage subclasses it
# to add the ``sum`` / ``mean`` reductions the losses need.
from dlfs import stage_import

Stage11_Tensor = stage_import("stage_11", "Tensor")


class Tensor(Stage11_Tensor):
    """The autodiff ``Tensor`` extended with ``sum`` / ``mean`` reductions.

    A reduction maps an array to fewer elements; its backward must re-expand the
    upstream grad back over the reduced axes. Build results via ``self._make_tensor``
    (== ``type(self)``) so a chain keeps returning stage_12 Tensors.
    """

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None,
            keepdims: bool = False) -> "Tensor":
        """Sum along ``axis`` (whole array when ``axis is None``). Returns a graph node.

        Forward: ``np.sum``, wrapped via ``self._make_tensor``.
        Backward: sum has local grad 1, so broadcast ``z.grad`` back to ``self.shape``
        (when ``keepdims=False``, restore the dropped axes as size-1 first), accumulate
        with ``+=``.

        ``x.sum()`` of ``[[1,2],[3,4]]`` -> ``10``; after backward ``x.grad`` is all 1s.
        """
        raise NotImplementedError("Tensor.sum")

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None,
             keepdims: bool = False) -> "Tensor":
        """Average along ``axis`` (whole array when ``axis is None``). The losses use
        this to keep their scale independent of batch size.

        Forward: ``np.mean``, wrapped via ``self._make_tensor``.
        Backward: ``mean = sum / N`` (N = count of reduced elements), so it's the
        ``sum`` backward divided by N: each input grad is ``1/N``.

        ``x.mean()`` of ``[[1,2],[3,4]]`` -> ``2.5``; after backward ``x.grad`` is all 1/4.
        """
        raise NotImplementedError("Tensor.mean")


def mse_loss(pred, target) -> "Tensor":
    """Mean squared error: scalar Tensor L = mean( (pred - target)**2 )."""
    # TODO: implement MSE using Tensor ops and .mean()
    raise NotImplementedError("mse_loss")


def mae_loss(pred, target) -> "Tensor":
    """Mean absolute error: scalar Tensor L = mean( |pred - target| ).

    Build abs from Tensor ops (e.g. |d| = relu(d) + relu(-d)); never hand-write grads.
    """
    # TODO: implement MAE using Tensor ops and .mean()
    raise NotImplementedError("mae_loss")


def log_softmax(logits) -> "Tensor":
    """Log-softmax of ONE logit vector: (C,) raw scores -> (C,) log-probs log(p_c).

    Computed stably with m = max_c z_c (a constant -- no grad through the shift):

        log_softmax(z)_c = (z_c - m) - log( sum_k exp(z_k - m) )

    Subtracting m keeps every exp argument <= 0 so it can't overflow, and m cancels, so
    the result is unchanged. (Writing log(softmax(z)) instead re-introduces the overflow.)

    Steps: m off the data (constant); shift = logits - m; lse = log(sum(exp(shift)));
    return shift - lse.
    """
    # TODO: implement the steps. Do NOT take the shortcut log(softmax(logits)).
    raise NotImplementedError("log_softmax")


def softmax(logits) -> "Tensor":
    """Softmax of ONE logit vector: (C,) -> (C,) probabilities (in (0,1), sum to 1).

    Since p = exp(log p), this is just exp of log_softmax -- reuse it and inherit its
    overflow-safety.
    """
    # TODO: exponentiate log_softmax(logits).
    raise NotImplementedError("softmax")


def cross_entropy_loss(logits, target) -> "Tensor":
    """Softmax cross-entropy for ONE example -> scalar loss.

    logits -- (C,) raw class scores.
    target -- the correct class, either an int t, or a (C,) distribution (1.0 at the
              true class for a hard label, or fractions for a soft one).

    The loss is the negative log-prob of the correct class, -log(p[t]) with
    p = softmax(logits): right & confident -> near 0, wrong & confident -> large.

    Build from log_softmax (stable; never log(softmax)) so backward() makes the gradient
    on its own. With lp = log_softmax(logits) and the target as a length-C vector y:

        L = -sum_c y_c * lp_c        (= -lp[t] for a hard label)

    Turning `target` into y is yours to design.
    """
    # TODO: lp = log_softmax(logits); express `target` as a length-C vector y;
    #       return -(lp * y).sum().
    raise NotImplementedError("cross_entropy_loss")
