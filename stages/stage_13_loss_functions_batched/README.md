# Stage 13: Loss functions over a batch

> This is the second of the two loss stages. [stage_12](../stage_12_loss_functions/) built
> softmax / log-softmax / cross-entropy for **one example** — a `(C,)` logit vector and a single label.
> Here we lift those three to a whole **batch**: a `(B, C)` matrix of logits, `B` examples stacked as
> rows. The per-row math is exactly the single-example math from stage_12; the only new idea is doing it
> independently per row and then **averaging** the per-example losses over the batch.

## Why batching changes almost nothing

A network processes many examples at once for speed, so its output is a `(B, C)` matrix: row `b` holds
the `C` class scores for example `b`. Softmax, log-softmax and cross-entropy are all defined **per
example**, so the batched versions just apply the stage_12 formula to each row on its own.

Mechanically, every reduction that used to run over "the `C` classes" of a single vector now runs over
the **class axis** `axis=1` of the matrix, and the per-row result is kept as a `(B, 1)` column so it
broadcasts back over the `C` classes of its row. Cross-entropy adds one extra step on top: after getting
one loss per row, it takes the **mean** over the `B` rows so the loss does not grow with batch size.

`mse_loss` and `mae_loss` already reduced over *all* elements with `.mean()`, so they work on batched
inputs unchanged — this stage simply re-exports them from stage_12.

## Softmax and log-softmax, per row

With a per-row max $m_b = \max_c z_{b,c}$ (a constant — no gradient flows through the shift):

$$p_{b,c} = \frac{e^{z_{b,c} - m_b}}{\sum_k e^{z_{b,k} - m_b}},
\qquad
\log\text{-softmax}(z)_{b,c} = (z_{b,c} - m_b) - \log\sum_k e^{z_{b,k} - m_b}.$$

The subtraction of $m_b$ is the same log-sum-exp stability trick as stage_12, now taken along `axis=1`
with `keepdims=True` so the `(B, 1)` column lines up with every row.

## Cross-entropy over the batch

For each row this is the stage_12 cross-entropy, $-\log p_{b,t_b}$. The batch loss averages those:

$$L = -\frac{1}{B}\sum_{b}\sum_{c} y_{b,c}\,\log p_{b,c},$$

where $y$ is the targets written as a `(B, C)` matrix (a one-hot row per integer label, or a soft-label
row given directly). Build it from `log_softmax`: multiply $y$ by the log-probs, sum over the class axis
to get one loss per row, then `.mean()` over the batch. As in stage_12 the gradient comes out clean —
now divided by the batch size:

$$\frac{\partial L}{\partial z_{b,c}} = \frac{p_{b,c} - y_{b,c}}{B}.$$

Let `backward()` reproduce it; do not hard-code the gradient.

## Cumulative build

Import the `Tensor`, `mse_loss` and `mae_loss` from `stage_12` via `dlfs.stage_import` (stage_12's
`Tensor` already carries the `sum` / `mean` reductions, including `axis` / `keepdims`). Re-export the two
regression losses unchanged, and write the batched `log_softmax`, `softmax`, and `cross_entropy_loss` on
top. No new `Tensor` ops are needed — only `axis=1` reductions you already have.

## Exercise

In `code.py`, implement the following as **functions** that take and return `Tensor`s. Build every loss
out of `Tensor` ops so gradients flow through `Tensor.backward()` — never hand-write a `.grad`. Allowed
tools: `numpy` (forward array construction only), the Python stdlib, and the `Tensor` engine. No PyTorch
/ autograd.

- `log_softmax(logits) -> Tensor`: `(B, C)` logits in, `(B, C)` of $\log p$ out, via stable log-sum-exp
  with the **per-row** max as a constant. Equals `logits - logsumexp(logits, axis=1, keepdims=True)`.
- `softmax(logits) -> Tensor`: `(B, C)` probabilities; each row sums to 1. May be `exp(log_softmax(...))`.
- `cross_entropy_loss(logits, targets) -> Tensor`: `logits` `(B, C)`; `targets` either a 1-D array of `B`
  integer class indices **or** a `(B, C)` one-hot / soft matrix. Return the scalar mean
  $-\frac{1}{B}\sum_b \log p_{b,t_b}$, built from `log_softmax` so the gradient is `(p - y)/B`.

## Done when

- `pytest stage_13_loss_functions_batched/test.py` passes.
- `softmax` rows each sum to 1 and match a NumPy reference; `log_softmax` equals `log(softmax(...))`; both
  are stable for logits like `[1000, 1001, 1002]` (no `inf` / `nan`).
- Cross-entropy forward matches the NumPy mean reference, accepts integer **and** one-hot targets
  identically, and is stable on large logits.
- Central-difference gradcheck of cross-entropy w.r.t. its logits matches the analytic `Tensor.grad`, and
  that gradient equals `(softmax(logits) - onehot(targets)) / B`.
