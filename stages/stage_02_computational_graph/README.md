# Stage 02: Computational graph

**Context** — In `stage_01` you built `Value`, which already *records* where each result came from: every operation stores its operands in `_prev` and its op label in `_op`. But nothing *reads* that bookkeeping yet — the graph exists in memory and is invisible. This stage makes the graph **first-class** by adding the part that uses it: a `trace(root)` walker that enumerates the whole DAG, and a graph-aware `__repr__` that surfaces `_op`. The arithmetic operators are inherited untouched.

**Background** — A scalar expression like $f = (a + b) \cdot c$ is really a tree of operations: a `+` node feeding a `*` node. To differentiate it automatically we must be able to walk that structure. stage_01 already stores it on each node: `_prev`, the **set of input `Value`s** that produced it; `_op`, a **string label** for the operation (`'+'`, `'*'`, `''` for leaves); and `_backward`, a **closure hook** that is a no-op for now (stage_03 installs the real per-op gradient rule on it). Leaves (numbers you create directly) have an empty `_prev` and empty `_op`. Following `_prev` from any node enumerates the subgraph that built it — which is exactly what `trace` does. `_prev` is a **set** so a reused operand (`a * a`) is a single parent and the DAG walk terminates; operand order is **not** stored, because the gradient closures added in stage_03 capture each operand directly — so `a - b` and `a / b` will know which side is which without the node remembering order. This stage adds no arithmetic and computes no gradients: forward values are unchanged from stage 01 and `grad` still defaults to `0.0`. The reverse pass that will use these edges relies on the chain rule, e.g. for $L$ depending on $u = a + b$,
$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial u}\cdot\frac{\partial u}{\partial a}, \qquad \frac{\partial u}{\partial a} = 1,$$
and for $v = a \cdot b$, $\frac{\partial v}{\partial a} = b$ and $\frac{\partial v}{\partial b} = a$. We do **not** compute gradients yet — we only build the graph those local derivatives will later flow through. The one structural rule that matters: building the graph must be acyclic. Reusing a node (e.g. `a * a`) is fine and must not duplicate it in `_prev` (hence a *set*).

**Watch**
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Karpathy builds exactly this `Value` graph; watch the first ~25 min where `_prev`/`_op` are introduced and the graph is visualized.
- [What is backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U) — 3Blue1Brown's mental model of computation flowing through a graph; motivates why we record structure now.

**Cumulative** — This stage imports `Value` via `stage_import("stage_01", "Value")` and SUBCLASSES it (not a rewrite). Because stage_01's operators already build their results through `_make` (constructing `type(self)(...)` and wiring `_prev`/`_op`), subclassing alone makes the whole DAG out of *this* stage's `Value` — so you do **not** override `__init__`, `__add__`, `__mul__`, or `__pow__`. It ADDS only: a graph-aware `__repr__`, and a module-level `trace(root)` to walk the DAG.

**Exercise** — Extend `Value` from `stage_01` in `code.py` so the graph is walkable. Add nothing that computes gradients yet.
- The constructor and all arithmetic (`__add__`/`__mul__`/`__pow__` and the derived/reflected ops) are INHERITED from stage_01 and already record `_prev`/`_op`. Do **not** re-implement them.
- `__repr__(self)`: return a repr that includes `data=<x>` and `op=<op>` (e.g. `Value(data=3.0, op='+')`) — exact formatting is free; the tests only check that `data=` and `op=` appear (surfaces the `_op` label).
- Provide a free function `trace(root)` that walks `_prev` from `root` and returns `(nodes, edges)`: `nodes` a set of all `Value`s reachable from `root`, `edges` a set of `(parent, child)` tuples, where the parent is the operand/input and the child is the result built from it. It must not revisit nodes (use a visited set) so it terminates even though the graph is a DAG.
- Forward values and graph metadata are exactly as stage 01 produces them; this stage only adds the ability to *read* the graph.
- Allowed tools: Python stdlib only. No NumPy needed here; no autodiff libraries.

**Done when**
- [ ] `pytest stage_02_computational_graph/test.py` passes.
- [ ] Leaves have `_prev == set()` and `_op == ''`; `+`/`*` results have the right two parents and correct `_op` (inherited from stage 01).
- [ ] Every node has a callable `_backward`; a leaf's is a no-op that changes no grad.
- [ ] `a * a` produces a node with a single parent in `_prev` (set dedup) and `trace` terminates with no duplicates.
- [ ] `trace(root)` returns every node and every parent→child edge exactly once.
