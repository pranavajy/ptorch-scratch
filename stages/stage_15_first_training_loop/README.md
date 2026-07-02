# Stage 15: First training loop

**Context** — Every prior stage built one piece in isolation: the autodiff `Tensor` engine (`stage_08`, broadcasting backward in `stage_11`, `sum`/`mean` reductions in `stage_12`), the `MLP` model (`stage_11`), `mse_loss` (`stage_12`), and the `SGD` optimizer (`stage_14`). None of them *learns* anything on its own. This stage wires them into the canonical training loop — the five lines at the heart of every deep-learning framework — and the accuracy metric you use to watch it learn. By the end you train an MLP to separate two interleaved half-moons, a problem no linear model can solve.

**Background** — Training is iterative descent on the loss surface. With parameters $\theta$ (the weight/bias `Tensor`s of every layer) and a scalar loss $L(\theta)$, each epoch repeats:

1. **forward** — $\hat y = f_\theta(X)$, then $L = \mathrm{mse}(\hat y, y)$;
2. **backward** — `L.backward()` fills every `p.grad` $= \partial L / \partial p$ via the autodiff engine;
3. **step** — $\theta \leftarrow \theta - \eta\, \partial L/\partial \theta$ (that is `optimizer.step()`);
4. **zero-grad** — reset every `p.grad` to zeros (`optimizer.zero_grad()`).

Step 4 is the loop's hygiene contract. Within one backward pass grads *accumulate* (`+=` — a tensor reused twice sums both contributions), and in PyTorch they also accumulate *across* `loss.backward()` calls, which is why explicit zeroing is part of every real training loop. Our `stage_08` engine happens to be more forgiving — `backward()` first resets the grad of every node reachable from the loss — so skipping `zero_grad` would not corrupt the step *here*; the contract you implement anyway is: **`train` ends every epoch (and therefore returns) with all parameter grads zeroed**, so the caller can immediately `backward()` something else, and later optimizer stages (momentum, Adam) can trust the state they read. The loss is on-graph (a `Tensor`, so it can backprop); **accuracy is off-graph**: a read-only metric computed from `.data` that must never create graph nodes or touch grads. For $\pm 1$ targets, a prediction is correct when $\mathrm{sign}(\hat y_i) = \mathrm{sign}(y_i)$.

One integration trap this stage forces you through: the model outputs shape `(N, 1)` while labels naturally come as `(N,)`. Subtract those and broadcasting silently yields `(N, N)` — the loss still computes, but it is garbage. `train` therefore takes `Tensor`s only and normalizes `y` to a `(N, 1)` column before the loop.

**Watch**
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Karpathy's final section *is* this loop: forward, `loss.backward()`, nudge params, **zero grad**, repeat; he shows the exact bug when you forget to zero.
- [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) — 3Blue1Brown: the geometric picture of stepping downhill that the loop implements.

**Cumulative framework** — This stage does **not** redefine the model/loss/optimizer: it imports `Tensor` (`stage_12`), `MLP` (`stage_11`), `mse_loss` (`stage_12`), and `SGD` (`stage_14`) via `dlfs.stage_import` and re-exports them under canonical names. The toy datasets (`make_moons`, `make_spiral`) live in this stage's `test.py` as fixtures, and the plot helper (`plot_history`) ships already implemented — they are plumbing, not the lesson. You add exactly two things: `accuracy` and `train`. Later stages import `accuracy` / `train` from here (`stage_19` builds mini-batching on top of this exact full-batch driver), so keep the contracts below precise.

**Exercise** — Implement `accuracy` and `train` in `code.py`. Use **only** the imported `stage_08`–`stage_14` framework, NumPy (forward array math only), and the stdlib. Do **not** compute any gradient by hand: every gradient comes from `Tensor.backward()`.

- **Provided** — `plot_history(history, path=None)` (in `code.py`) plots the loss and accuracy curves `train` returns. The dataset fixtures `make_moons(n=200, noise=0.1, seed=None)` and `make_spiral(n_per_class=100, n_classes=2, noise=0.2, seed=None)` (in `test.py`) return `(X, y)` as `Tensor`s: `X` of shape `(N, 2)`, `y` of shape `(N, 1)` with values in `{-1.0, +1.0}`, deterministic given `seed` — that `(N, 1)` column is what your `train` is fed.
- `accuracy(pred, y) -> float`: fraction of examples where `sign(pred) == sign(y)`. Accepts `Tensor` or `ndarray` for either argument, and `(N, 1)` / `(N,)` shapes interchangeably (compare flattened); raise `ValueError` when the two carry different numbers of elements. Read-only and off-graph: work on `.data`, never build `Tensor` ops. A prediction of exactly `0.0` matches neither class and counts as wrong.
- `train(model, X, y, *, lr=0.1, epochs=200, optimizer=None) -> dict`: the full-batch loop. Contract:
  - `X` and `y` must be `Tensor`s — raise `TypeError` otherwise (this is what protects you from the `(N, N)` broadcast trap above).
  - `X` must be 2-D `(N, n_in)`; `y` must be `(N, 1)` or `(N,)` (normalize to a `(N, 1)` column **off-graph**, e.g. `Tensor(y.data.reshape(-1, 1))`); raise `ValueError` on any other shape or when the row counts disagree.
  - If `optimizer is None`, build `SGD(model.parameters(), lr)`; when an optimizer IS passed, use it and ignore `lr`.
  - Each epoch: forward → `mse_loss` → `loss.backward()` → `optimizer.step()` → `optimizer.zero_grad()`, then record `float(loss.data)` and `accuracy(pred, y)` computed from that same forward pass.
  - Return `{"loss": [...], "accuracy": [...]}`, one float per epoch. After `train` returns, every parameter's `.grad` is all-zeros (the loop ends on `zero_grad`), so the caller can immediately `backward()` something else.
- Allowed tools: `numpy`, `matplotlib` (plot helper only), Python stdlib, and the imported `stage_08`–`stage_14` code. **No** PyTorch/autograd/etc.

**Done when**
- `pytest stages/stage_15_first_training_loop/test.py` passes.
- `train` run for a handful of epochs is *exactly* equal (loss history and final parameters) to a hand-rolled forward/loss/backward/step/zero loop on an identically-seeded model — this equality catches a shuffled step order or any stray extra update; a separate test pins that `train` leaves every `p.grad` zeroed.
- On noiseless moons the MLP reaches ≥ 90% training accuracy within a few hundred epochs, and the recorded accuracy history ends where `accuracy(model(X), y)` says it should.
- `train` rejects raw ndarrays with `TypeError` and mismatched shapes with `ValueError`; `y` passed as `(N,)` trains identically to the same `y` passed as `(N, 1)`.
