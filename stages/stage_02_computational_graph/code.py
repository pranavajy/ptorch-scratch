"""Stage 2: Computational graph.

Subclass stage_01 `Value` to make its operation graph first-class: every result
node remembers the parents it was built from (`_prev`), the op that built it
(`_op`), and reserves a `_backward` hook (a no-op for now; stage_03 fills it in).
A `trace(root)` walker enumerates the DAG. No gradients computed here; forward
arithmetic stays identical to stage_01. Allowed tools: Python standard library ONLY.
"""

from dlfs import stage_import

# stage_01 Value (extend, do not rewrite)
Stage1_Value = stage_import("stage_01", "Value")


class Value(Stage1_Value):
    """stage_01 `Value`, extended so its operation graph is first-class.

    Each result node records:
    - ``_prev``: the **set** of parent ``Value``s it was built from. A set so a
      reused operand (``a * a``) is a single parent and the DAG walk terminates.
    - ``_op``: a string label for the op (``'+'``, ``'*'``, ``''`` for leaves).
    - ``_backward``: a no-op closure here; stage_03 installs the real per-op
      gradient rule on it. Reserving the field now keeps the field set stable.

    Operand ORDER is **not** stored on the node: a set is unordered, and that is
    fine because the gradient rules added in stage_03 are closures that capture
    each operand directly (so ``a - b`` and ``a / b`` know which side is which
    without the node remembering order). Leaves have ``_prev == set()``,
    ``_op == ''``.
    """

    def __init__(self, data, _children=(), _op=""):
        """Init stage_01 fields, then record graph provenance.

        Call ``Stage1_Value.__init__(self, data)`` for ``data``/``grad``, then set
        ``self._prev = set(_children)``, ``self._op = _op``, and install a no-op
        ``self._backward = lambda: None`` (stage_03 overrides it per op).
        """
        # TODO: super().__init__(data); then self._prev=set(_children); self._op=_op;
        # self._backward = lambda: None
        raise NotImplementedError("stage_02: implement Value.__init__")

    def __add__(self, other):
        """Return self + other as a Value recording this addition (_op='+').

        Coerce a number operand to ``Value(other)``; build
        ``Value(self.data + other.data, (self, other), '+')``.
        """
        # TODO: implement graph-recording addition (coerce numbers to Value)
        raise NotImplementedError("stage_02: implement Value.__add__")

    def __mul__(self, other):
        """Return self * other as a Value recording this multiply (_op='*').

        Coerce a number operand to ``Value(other)``; build
        ``Value(self.data * other.data, (self, other), '*')``.
        """
        # TODO: implement graph-recording multiply (coerce numbers to Value)
        raise NotImplementedError("stage_02: implement Value.__mul__")

    def __repr__(self):
        """Graph-aware debug string, e.g. ``Value(data=3.0, op='+')``."""
        # TODO: implement repr surfacing data and _op
        raise NotImplementedError("stage_02: implement Value.__repr__")


def trace(root):
    """Walk the graph backward from `root`, returning (nodes, edges).

    nodes: set of every Value reachable from root (including root).
    edges: set of (parent, child) tuples, one per _prev link.
    Use a visited set so the DAG walk terminates on reused nodes (e.g. a * a).
    """
    # TODO: implement the backward graph walk
    raise NotImplementedError("stage_02: implement trace")
