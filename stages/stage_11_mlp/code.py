"""Stage 12: MLP + broadcasting-aware Tensor.

Two things ship here:

* ``MLP`` -- a multilayer perceptron: a chain of pure-linear ``Dense`` layers
  (stage_10) with a nonlinearity applied between them, built on the autodiff
  ``Tensor`` (stage_08).
* ``Tensor`` -- a thin subclass of the stage_08 ``Tensor`` that finally adds the
  **broadcasting-correct backward** stage_08 deferred to here.  The stage_08
  engine restricts elementwise binary ops to EQUAL-SHAPED operands; this stage
  overrides ``__add__``/``__mul__`` so differently-shaped-but-broadcastable
  operands forward via NumPy broadcasting AND each parent's gradient is reduced
  (the "unbroadcast" rule) back to that parent's original shape.  stage_12's
  stable softmax (``logits - logsumexp(..., keepdims=True)``, i.e. ``(B,C)-(B,1)``)
  relies on this.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

# Building blocks from earlier stages (re-exported for tests / later stages).
from dlfs import stage_import

Stage8_Tensor = stage_import("stage_08", "Tensor")
Stage10_Dense = stage_import("stage_10", "Dense")

# An operand the broadcasting binary ops accept: a Tensor, or a raw
# number/array we wrap (mirrors stage_08's ``Operand`` alias).
Operand = Union["Tensor", float, int, np.ndarray, list]


class Tensor(Stage8_Tensor):
    """The stage_08 autodiff ``Tensor`` extended with broadcasting backward.

    stage_08 keeps elementwise binary ops to equal-shaped operands and defers
    "general broadcasting gradient reduction to stage_11" -- that is this class.
    It overrides the elementwise binary ops so that when operands have DIFFERENT
    but broadcastable shapes, the forward uses NumPy broadcasting and each
    parent's incoming gradient is *unbroadcast* (summed over the axes that were
    broadcast, then reshaped) back to that parent's original shape.

    The non-broadcasting primitives (``__pow__``, ``relu``, ``tanh``, ``exp``,
    ``log``, ``@``) are inherited unchanged from stage_08; the derived ops
    (``__sub__``, ``__truediv__``, ``__neg__`` and the reflected forms) compose
    out of the overridden ``__add__``/``__mul__`` and so become broadcasting too
    without any new ``_backward``.
    """

    @staticmethod
    def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """Reduce a broadcasted gradient back to an operand's original ``shape``.

        When ``z = f(x, y)`` forwards through NumPy broadcasting, the upstream
        gradient ``grad`` has the *broadcast* shape, not ``x.shape``.  The chain
        rule says a value that was copied across a broadcast axis receives the
        SUM of the gradient over the copies.  So to send ``grad`` back to a
        parent whose original shape is ``shape`` we must:

          1. Sum away any EXTRA LEADING axes ``grad`` has beyond ``len(shape)``
             (these came from rank promotion, e.g. ``(3,)`` -> ``(2,3)``), then
          2. For every axis where the operand had size 1 but the broadcast shape
             had size > 1, sum ``grad`` over that axis with ``keepdims=True``
             (the size-1 axis was stretched), and finally
          3. ``reshape`` to exactly ``shape`` (a no-op once 1 & 2 are done).

        The result is ``np.ndarray`` of shape ``shape``, ready to ``+=`` into
        that parent's ``.grad``.

        Example: ``z = a + b`` with ``a.shape == (2, 3)``, ``b.shape == (3,)``
        and upstream ``grad`` of shape ``(2, 3)``.  ``a`` gets ``grad`` as-is;
        ``b`` gets ``grad.sum(axis=0)`` -> shape ``(3,)`` (the column sums).
        """
        # TODO: implement the unbroadcast reduction described above:
        #   - while grad.ndim > len(shape): grad = grad.sum(axis=0)
        #   - for each axis i where shape[i] == 1 and grad.shape[i] > 1:
        #         grad = grad.sum(axis=i, keepdims=True)
        #   - return grad.reshape(shape)
        raise NotImplementedError("Tensor._unbroadcast")

    def __add__(self, other: "Operand") -> "Tensor":
        """Broadcasting elementwise add: ``z = self + other``.

        Forward via NumPy broadcasting (``self.data + other.data``); in
        ``_backward`` push ``self._unbroadcast(out.grad, self.shape)`` to ``self``
        and ``self._unbroadcast(out.grad, other.shape)`` to ``other`` so each
        parent's grad is reduced back to its own shape.  Equal-shaped operands
        reduce to the stage_08 behaviour (unbroadcast is then a no-op)."""
        # TODO: coerce other; out = Tensor(self.data + other.data, (self, other), "+");
        #       _backward: each parent.grad += _unbroadcast(out.grad, parent.shape).
        raise NotImplementedError("Tensor.__add__")

    def __mul__(self, other: "Operand") -> "Tensor":
        """Broadcasting elementwise multiply: ``z = self * other``.

        Forward via NumPy broadcasting (``self.data * other.data``); local grads
        are ``g * other`` for ``self`` and ``g * self`` for ``other`` (each
        evaluated at the BROADCAST shape), then unbroadcast back to each parent's
        original shape before accumulating."""
        # TODO: coerce other; out = Tensor(self.data * other.data, (self, other), "*");
        #       _backward: self.grad  += _unbroadcast(out.grad * other.data, self.shape)
        #                  other.grad += _unbroadcast(out.grad * self.data,  other.shape).
        raise NotImplementedError("Tensor.__mul__")


# ``Tensor`` (above) is this stage's public broadcasting-capable autodiff node.
# stage_12's stable softmax needs ``(B,C) - (B,1)`` to backprop correctly; that
# broadcasting backward is delivered HERE, satisfying stage_12's reliance even
# though stage_12 keeps importing the engine via ``stage_import("stage_08",
# "Tensor")`` (its construction sites just feed differently-shaped operands).


class MLP:
    """A multilayer perceptron: ``Dense`` layers + activations. sizes
    ``[n_in, ..., n_out]`` builds len(sizes)-1 Dense layers; activation follows
    each hidden layer, out_activation the last (each in {"tanh","relu","none"})."""

    def __init__(
        self,
        sizes: Sequence[int],
        activation: str = "tanh",
        out_activation: str = "none",
        seed: Optional[int] = None,
    ) -> None:
        # TODO: validate args; build the Dense layers (per-layer derived seeds).
        raise NotImplementedError("MLP.__init__")

    @staticmethod
    def _apply_activation(z: "Stage8_Tensor", name: str) -> "Stage8_Tensor":
        """Apply named pointwise activation via the Tensor's own methods; raise on unknown name."""
        # TODO: dispatch "none"/"tanh"/"relu" to z / z.tanh() / z.relu().
        raise NotImplementedError("MLP._apply_activation")

    def forward(self, x) -> "Stage8_Tensor":
        """Chain layers, applying activation after each (out_activation after the
        last). x ``(n_in,)`` or ``(batch, n_in)`` -> ``(n_out,)`` / ``(batch, n_out)``."""
        # TODO: chain layers with the right activation per layer.
        raise NotImplementedError("MLP.forward")

    def __call__(self, x) -> "Stage8_Tensor":
        """Alias for :meth:`forward`."""
        # TODO: delegate to forward.
        raise NotImplementedError("MLP.__call__")

    def parameters(self) -> List["Stage8_Tensor"]:
        """Return every learnable parameter from every layer, flattened in layer order."""
        # TODO: flatten each layer's parameters().
        raise NotImplementedError("MLP.parameters")

    def zero_grad(self) -> None:
        """Reset the gradient of every parameter to zeros."""
        # TODO: zero each parameter's grad.
        raise NotImplementedError("MLP.zero_grad")

    def __repr__(self) -> str:
        # TODO: summarize sizes and activations.
        raise NotImplementedError("MLP.__repr__")
