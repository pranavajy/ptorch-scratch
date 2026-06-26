"""Stage 09: Neuron.

A single neuron ``y = phi(x @ w + b)`` built on stage_08's autodiff ``Tensor``.
Wire up the forward expression only; gradients flow through ``Tensor.backward()``.
"""

from __future__ import annotations

import numpy as np

from dlfs import stage_import

# Tensor engine from stage_08, re-exported as the public ``Tensor``.
Stage8_Tensor = stage_import("stage_08", "Tensor")
Tensor = Stage8_Tensor


class Neuron:
    """A single neuron: ``y = phi(x @ w + b)``, built on stage_08's ``Tensor``."""

    def __init__(self, n_in: int, activation: str = "tanh", seed: int | None = None):
        """Construct leaf params: w shape (n_in,), scalar bias b=0; activation in {tanh, relu, none}."""
        # TODO: validate activation, build RNG, init w/b as leaf Tensors, store config
        raise NotImplementedError("Neuron.__init__")

    def __call__(self, x) -> "Stage8_Tensor":
        """Forward pass: z = x @ w + b then phi(z). x shape (n_in,) or (batch, n_in)."""
        # TODO: coerce x to Tensor, compute affine map, apply activation
        raise NotImplementedError("Neuron.__call__")

    def parameters(self) -> list:
        """Return the learnable parameters as ``[self.w, self.b]``."""
        raise NotImplementedError("Neuron.parameters")

    def zero_grad(self) -> None:
        """Reset the gradient of every parameter to zeros."""
        raise NotImplementedError("Neuron.zero_grad")

    def __repr__(self) -> str:
        """e.g. ``Neuron(n_in=3, activation='tanh')``."""
        raise NotImplementedError("Neuron.__repr__")
