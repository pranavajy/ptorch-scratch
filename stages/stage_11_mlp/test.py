"""Tests for Stage 11: MLP.

These tests verify the forward shapes of a multilayer perceptron, that it
gathers all of its layers' parameters, and they gradient-check a scalar loss
(the sum of the network output) with respect to every layer's weights and bias
and with respect to the input, using central differences::

    df/dp ~= (f(p + eps) - f(p - eps)) / (2 * eps)

compared against the analytical gradients produced by ``Tensor.backward()`` from
stage_08 (propagated through the ``Dense`` layers from stage_10). ``MLP`` lives in
this stage's ``code.py`` and is built on the ``Dense`` (stage_10) and ``Tensor``
(stage_08) classes imported through ``dlfs.stage_import``. If any earlier stage
is not yet implemented, the suite skips rather than erroring, so you can run it
incrementally.

Run with:  pytest stage_11_mlp/test.py
"""
import os as _os
import sys as _sys

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
# Make the shared `dlfs` shim importable (it lives at the curriculum root).
sys.path.insert(0, _ROOT)

# --- Import the things under test, skipping cleanly if not ready yet. --------
# `code.py` binds `Stage8_Tensor` (stage_08) and `Stage10_Dense` (stage_10) via
# dlfs.stage_import and defines this stage's `MLP` on top of them.
try:
    # --- resolve sibling code.py (avoid stdlib `code` collision) ---
    import importlib.util as _ilu
    _THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
    _ROOT = _os.path.dirname(_THIS_DIR)
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)
    _spec = _ilu.spec_from_file_location(
        "code", _os.path.join(_THIS_DIR, "code.py")
    )
    _mod = _ilu.module_from_spec(_spec)
    _sys.modules["code"] = _mod
    _spec.loader.exec_module(_mod)
    from code import MLP, Stage10_Dense, Stage8_Tensor, Tensor
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_11 MLP / stage_10 Dense / stage_08 Tensor not importable yet: {exc}",
        allow_module_level=True,
    )

EPS = 1e-6
ATOL = 1e-6
RTOL = 1e-4


# --- Small helpers -----------------------------------------------------------
def as_array(t):
    """Return the underlying numpy array of a Tensor (or pass arrays through)."""
    return np.asarray(t.data if hasattr(t, "data") else t, dtype=float)


def make_tensor(arr, requires_grad=True):
    """Build a Tensor from a numpy array, tolerating different ctor kwargs."""
    arr = np.asarray(arr, dtype=float)
    try:
        return Stage8_Tensor(arr, requires_grad=requires_grad)
    except TypeError:
        return Stage8_Tensor(arr)


def scalar_out(t):
    """Reduce a Tensor's output to a python float for finite-diff probing."""
    return float(np.sum(as_array(t)))


def central_diff(f, x, eps=EPS):
    """Numerical gradient of scalar-valued f at numpy point x (any shape)."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = f(x)
        x[idx] = orig - eps
        fm = f(x)
        x[idx] = orig
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def build_net(sizes, activation="tanh", out_activation="none", seed=0):
    return MLP(sizes, activation=activation, out_activation=out_activation, seed=seed)


# --- Construction & structure ------------------------------------------------
def test_layer_count_and_widths():
    net = build_net([3, 5, 4, 2], seed=0)
    assert len(net.layers) == 3, "an MLP over [3,5,4,2] must have 3 Dense layers"


def test_every_layer_is_a_dense():
    net = build_net([4, 8, 1], seed=0)
    for layer in net.layers:
        assert isinstance(layer, Stage10_Dense), "each MLP layer must be a stage_10 Dense"


def test_parameters_are_collected_from_all_layers():
    net = build_net([3, 6, 6, 2], seed=1)
    params = net.parameters()
    # 3 Dense layers, 2 params (weight + bias) each -> 6 parameter tensors.
    assert len(params) == 6, "parameters() must flatten all layers' params (2 per layer)"
    expected = [p for layer in net.layers for p in layer.parameters()]
    assert len(params) == len(expected)
    # Identity check: the same tensor objects, not copies.
    for p, q in zip(params, expected):
        assert p is q, "parameters() must return the actual layer parameter tensors"


def test_repr_mentions_sizes_and_activation():
    net = build_net([2, 16, 1], activation="tanh", out_activation="none", seed=0)
    r = repr(net)
    assert "MLP" in r and "tanh" in r


def test_reproducible_with_seed():
    a = build_net([3, 7, 2], seed=123)
    b = build_net([3, 7, 2], seed=123)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert np.allclose(as_array(pa), as_array(pb)), "same seed -> same weights"


def test_distinct_layers_have_distinct_weights():
    # Per-layer seed derivation should make the two weight matrices differ
    # (they also have different shapes here, but check the values are not a
    # trivially-shared object).
    net = build_net([4, 4, 4], seed=0)
    w0 = as_array(net.layers[0].parameters()[0])
    w1 = as_array(net.layers[1].parameters()[0])
    assert not np.allclose(w0, w1), "different layers should not share identical weights"


# --- Forward shapes ----------------------------------------------------------
def test_single_input_shape():
    net = build_net([3, 5, 2], seed=2)
    y = net(make_tensor([0.5, -1.0, 2.0], requires_grad=False))
    assert as_array(y).shape == (2,), "(n_in,) input must yield (n_out,) output"


def test_batched_input_shape():
    net = build_net([3, 5, 2], seed=2)
    X = make_tensor([[0.5, -1.0, 2.0], [1.0, 0.0, -0.5]], requires_grad=False)
    y = net(X)
    assert as_array(y).shape == (2, 2), "(batch, n_in) input must yield (batch, n_out)"


def test_call_is_forward():
    net = build_net([2, 4, 3], seed=5)
    x = make_tensor([0.3, -0.7], requires_grad=False)
    y1 = net(x)
    y2 = net.forward(make_tensor([0.3, -0.7], requires_grad=False))
    assert np.allclose(as_array(y1), as_array(y2)), "__call__ must equal forward()"


# --- Depth + nonlinearity actually matters -----------------------------------
def test_hidden_nonlinearity_is_not_affine():
    """A 1->H->1 MLP with a tanh hidden layer must NOT be an affine function of x.

    For an affine map g, the second difference g(x+h) - 2 g(x) + g(x-h) is zero.
    A genuine nonlinearity makes it nonzero.
    """
    net = build_net([1, 8, 1], activation="tanh", out_activation="none", seed=11)

    def g(xv):
        return scalar_out(net(make_tensor(np.array([xv]), requires_grad=False)))

    h = 0.5
    x0 = 0.3
    second_diff = g(x0 + h) - 2.0 * g(x0) + g(x0 - h)
    assert abs(second_diff) > 1e-6, (
        "MLP with a hidden nonlinearity must not collapse to an affine map"
    )


# --- Gradient checks: scalar loss w.r.t. every parameter ---------------------
@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_gradcheck_wrt_all_params(activation):
    sizes = [3, 5, 4, 2]
    net = build_net(sizes, activation=activation, out_activation="none", seed=7)
    x_np = np.array([0.7, -1.3, 0.2])

    # Analytical gradients via one backward pass on sum(output).
    net.zero_grad()
    out = net(make_tensor(x_np, requires_grad=False))
    loss = out.sum() if hasattr(out, "sum") else out
    # If Tensor lacks .sum(), backward() on the (n_out,) tensor seeds ones,
    # which equals d(sum(out)). Either path matches our scalar_out finite diff.
    loss.backward()

    params = net.parameters()
    analytic = [as_array(p.grad).copy() for p in params]

    saved = [as_array(p).copy() for p in params]

    def f_factory(k):
        def f(pv):
            params[k].data = pv.copy().reshape(saved[k].shape)
            val = scalar_out(net(make_tensor(x_np, requires_grad=False)))
            params[k].data = saved[k].copy()
            return val
        return f

    for k, p in enumerate(params):
        g_num = central_diff(f_factory(k), saved[k].copy())
        assert np.allclose(analytic[k], g_num, atol=ATOL, rtol=RTOL), (
            f"[{activation}] dLoss/dparam[{k}] (shape {saved[k].shape}) mismatch:\n"
            f" analytic=\n{analytic[k]}\n numeric =\n{g_num}"
        )


# --- Gradient check: scalar loss w.r.t. the input ----------------------------
@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_gradcheck_wrt_input(activation):
    net = build_net([4, 6, 3], activation=activation, out_activation="none", seed=9)
    x_np = np.array([0.3, -0.6, 1.2, -2.0])

    net.zero_grad()
    x = make_tensor(x_np, requires_grad=True)
    out = net(x)
    loss = out.sum() if hasattr(out, "sum") else out
    loss.backward()
    g_analytic = as_array(x.grad)

    def f(xv):
        return scalar_out(net(make_tensor(xv, requires_grad=False)))

    g_num = central_diff(f, x_np.copy())
    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"[{activation}] dLoss/dx mismatch:\n analytic={g_analytic}\n numeric ={g_num}"
    )


# --- Gradient check over a batch (sum of all outputs) ------------------------
def test_gradcheck_batch_wrt_first_layer():
    net = build_net([3, 5, 2], activation="tanh", out_activation="none", seed=6)
    X = np.array([[0.5, -1.0, 2.0], [1.0, 0.5, -0.5], [-0.3, 0.8, 1.1]])

    net.zero_grad()
    out = net(make_tensor(X, requires_grad=False))
    loss = out.sum() if hasattr(out, "sum") else out
    loss.backward()
    W1 = net.layers[0].parameters()[0]
    g_analytic = as_array(W1.grad).copy()

    saved = as_array(W1).copy()

    def f(w):
        W1.data = w.copy().reshape(saved.shape)
        val = scalar_out(net(make_tensor(X, requires_grad=False)))
        W1.data = saved.copy()
        return val

    g_num = central_diff(f, saved.copy())
    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"batched dLoss/dW1 mismatch:\n analytic=\n{g_analytic}\n numeric =\n{g_num}"
    )


# --- zero_grad clears accumulated gradients across all layers -----------------
def test_zero_grad_clears_every_param():
    net = build_net([3, 4, 2], activation="tanh", seed=8)
    out = net(make_tensor([1.0, 2.0, 3.0], requires_grad=False))
    loss = out.sum() if hasattr(out, "sum") else out
    loss.backward()
    assert any(np.any(as_array(p.grad) != 0.0) for p in net.parameters()), (
        "some gradient should be populated after backward"
    )
    net.zero_grad()
    for p in net.parameters():
        assert np.allclose(as_array(p.grad), 0.0), "zero_grad must clear every param grad"


# --- Broadcasting backward (the stage_11 Tensor subclass) --------------------
# stage_08's Tensor only allows equal-shaped elementwise operands; this stage's
# `Tensor` subclass adds broadcasting forward + unbroadcast backward, which
# stage_12's stable softmax (`(B,C) - (B,1)`) and bias-row broadcasting need.
def bcast_tensor(arr, requires_grad=True):
    """Build a *broadcasting* Tensor (the stage_11 subclass), tolerating ctors."""
    arr = np.asarray(arr, dtype=float)
    try:
        return Tensor(arr, requires_grad=requires_grad)
    except TypeError:
        return Tensor(arr)


def test_tensor_is_broadcasting_subclass():
    """The exported `Tensor` must be the stage_08 engine extended in this stage."""
    assert issubclass(Tensor, Stage8_Tensor), (
        "stage_11 must export a Tensor that subclasses the stage_08 Tensor"
    )


def test_broadcast_add_row_vector_forward_and_grad():
    """(2,3) + (3,): forward equals numpy broadcast; (3,) grad is the column
    sums (shape (3,)) and (2,3) grad is ones (seed=ones)."""
    a_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    b_np = np.array([10.0, 20.0, 30.0])                   # (3,)
    a = bcast_tensor(a_np)
    b = bcast_tensor(b_np)

    z = a + b
    # Forward matches numpy broadcasting.
    assert as_array(z).shape == (2, 3)
    assert np.allclose(as_array(z), a_np + b_np)

    z.backward()  # seeds grad = ones_like(z), i.e. all-ones (2,3)
    # d/da of sum(a + b) is ones at a's shape; d/db is the column-sums.
    assert as_array(a.grad).shape == (2, 3)
    assert np.allclose(as_array(a.grad), np.ones((2, 3)))
    assert as_array(b.grad).shape == (3,), "broadcast (3,) operand grad must keep shape (3,)"
    assert np.allclose(as_array(b.grad), np.ones((2, 3)).sum(axis=0))  # [2, 2, 2]


def test_broadcast_sub_keepdims_column_forward_and_grad():
    """(B,C) - (B,1): forward broadcasts the (B,1) column; the (B,1) operand's
    grad sums across the C axis (shape (B,1)), as stage_12 softmax needs."""
    B, C = 4, 3
    x_np = np.arange(B * C, dtype=float).reshape(B, C)  # (B, C)
    m_np = np.array([[0.5], [1.0], [-2.0], [3.0]])      # (B, 1)
    x = bcast_tensor(x_np)
    m = bcast_tensor(m_np)

    z = x - m
    assert as_array(z).shape == (B, C)
    assert np.allclose(as_array(z), x_np - m_np)

    z.backward()  # seed ones (B, C)
    # d/dx of sum(x - m) is ones (B, C); d/dm is -1 summed across C -> (B, 1).
    assert as_array(x.grad).shape == (B, C)
    assert np.allclose(as_array(x.grad), np.ones((B, C)))
    assert as_array(m.grad).shape == (B, 1), "broadcast (B,1) operand grad must keep shape (B,1)"
    assert np.allclose(as_array(m.grad), -np.ones((B, C)).sum(axis=1, keepdims=True))  # all -C


def test_broadcast_mul_scales_grad_by_other_operand():
    """(2,3) * (3,): multiply backward applies the local factor at the broadcast
    shape, then unbroadcasts -- the (3,) operand's grad is the column-sums of the
    OTHER operand's data."""
    a_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    b_np = np.array([2.0, 3.0, 4.0])                      # (3,)
    a = bcast_tensor(a_np)
    b = bcast_tensor(b_np)

    z = a * b
    assert np.allclose(as_array(z), a_np * b_np)

    z.backward()  # seed ones (2, 3)
    # d/da = ones * b broadcast -> b tiled over rows; d/db = (ones * a) summed over rows.
    assert np.allclose(as_array(a.grad), np.broadcast_to(b_np, (2, 3)))
    assert as_array(b.grad).shape == (3,)
    assert np.allclose(as_array(b.grad), a_np.sum(axis=0))  # [5, 7, 9]
