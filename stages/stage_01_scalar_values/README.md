# Stage 01: Scalar values & arithmetic

**Cumulative chain** — This is the ORIGIN stage: it imports nothing from earlier stages (there are none). It exports the base `Value`, which every later stage extends via `dlfs.stage_import` — stage 02 adds the `trace` walk over the graph, stage 03 installs the per-op `_backward` closures, and stage 04 adds the `.backward()` reverse pass. Crucially, this stage gives `Value` its **final shape** so those later stages only ever ADD behavior, never rewrite these operators.

**Context** — This is the first stage of the whole curriculum and the seed of the autodiff engine you will build. You wrap a single floating-point number in a `Value` object and teach it to do arithmetic (`+`, `-`, `*`, `/`) through Python operator overloading. There are **no gradients yet** — that machinery (`.backward()` filling `grad`) arrives in stage 04. The class already carries the graph fields (`_prev`/`_op`/`_backward`) and every result already records its operands, but those fields are **inert plumbing** here: nothing reads them yet. Reserving them now — instead of bolting them on later — is what lets every later stage subclass `Value` and *add* without rewriting the arithmetic. Here you build the object, the forward math, and that plumbing, plus the conceptual foundation of variables, functions, and derivatives.

Stages 01-05 reimplement Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) `Value` engine from scratch, one concept per stage; this stage is the `Value(data)` starting point.

**Background** — A neural network is just one enormous differentiable function built by composing tiny operations. To differentiate it automatically, we first need a data type we control at every step, instead of bare Python floats. A `Value` stores one number in `self.data`. Overloading `__add__`, `__mul__`, etc. lets `a + b` and `a * b` return *new* `Value`s, so expressions like `d = a * b + c` compose into the forward computation. Each result also records the operands it came from (`_prev`) and the op that built it (`_op`) — the skeleton of the computational graph — even though nothing walks that graph until stage 02. Two design choices make later stages purely additive: the constructor takes its final signature `(data, _children=(), _op="")` from day one, and every operator builds its result through a `_make` helper that constructs `type(self)(...)` rather than a hard-coded `Value(...)`. Because of `type(self)`, when stage 02+ subclass `Value`, the inherited operators automatically return the *subclass* and record the graph — no operator needs to be rewritten. The derivative is the foundation: for a function $f$, the derivative measures sensitivity of the output to a tiny change in an input,

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}.
$$

We approximate it numerically with the symmetric **central difference**, which has error $O(h^2)$ instead of $O(h)$:

$$
f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}.
$$

You will *not* derive analytic gradients in this stage — you only verify, numerically, that derivatives of your `Value` expressions exist and behave (e.g. $\frac{d}{da}(a b) = b$). Everything later (the computational graph in stage 02, the `.backward()` autodiff engine completed by stage 04) builds on the class and forward ops you write here.

**Watch**

- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Karpathy builds exactly this `Value` object from scratch; watch the first ~25 min (the class, `data`, and operator overloading) and stop before gradients.
- [Derivatives, the limit definition (Essence of Calculus, ch. 2)](https://www.youtube.com/watch?v=9vKqVkMQHKk) — 3Blue1Brown on what a derivative actually is, the intuition you will need from stage 02 onward.

**Exercise** — Implement the base `Value` class in `code.py` (this is the origin: no `dlfs.stage_import`).

- `__init__(self, data, _children=(), _op="")`: store the number in `self.data` (coerce to `float`), `self.grad = 0.0`, `self._prev = set(_children)`, `self._op = _op`, and a no-op `self._backward = lambda: None`. The graph fields are reserved now and read by later stages.
- `_make(self, data, _children, _op)`: return `type(self)(data, _children, _op)`. Every operator builds its result through this helper so that subclasses get returned automatically — this is what removes the operator rewrites in later stages.
- `__repr__`: return `Value(data=<x>)`.
- Implement, each returning a **new** `Value` (built via `_make`) holding the resulting number and recording its operands/op:
  - `__add__(self, other)` → `self.data + other.data`, `_op='+'`, `_children=(self, other)`
  - `__mul__(self, other)` → `self.data * other.data`, `_op='*'`, `_children=(self, other)`
  - `__pow__(self, exponent)` → `self.data ** exponent`, `_op=f'**{exponent}'`, `_children=(self,)`; `exponent` is a Python `int`/`float` (not a `Value`)
  - `__neg__(self)` → implement via `self * -1`
  - `__sub__(self, other)` → implement as `self + (-other)`
  - `__truediv__(self, other)` → implement as `self * other ** -1`
- Support mixing with plain numbers (e.g. `2 * a`, `a + 1`, `a - 3`): in `__add__`/`__mul__`, wrap a non-`Value` operand as `type(self)(other)` (so a coerced number is a node of the right class), and define the reflected ops `__radd__`, `__rmul__`, `__rsub__`, `__rtruediv__`. Keep the derived ops (`__sub__`/`__truediv__`/`__neg__` and the reflected ops) composed purely out of `+`/`*`/`**` so they route through the dunders and dispatch to subclasses.
- Allowed tools: Python stdlib only. **No** NumPy, no autodiff libraries. Do not add a `.backward()` and do not compute any gradient — `_backward` stays a no-op and `grad` stays `0.0` here; those arrive in stages 03-04.
- Acceptance: arithmetic on `Value`s matches the same arithmetic on raw floats; operations with ints/floats on either side work; result nodes record `_prev`/`_op`.

**Done when**

- [ ] `pytest stage_01_scalar_values/test.py` passes.
- [ ] `Value` supports `+ - * / ** neg` and mixed int/float operands on both sides.
- [ ] The central-difference test confirms numerical derivatives of `Value` expressions exist and match the expected slope within tolerance.
