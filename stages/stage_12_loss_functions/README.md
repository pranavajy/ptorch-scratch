# Stage 12: Loss functions (a single example)

> This is the first of two loss stages. Here every loss is for **one example at a time** — no batch
> axis — so the shapes stay small and each loss is a single scalar you can reason about directly.
> [stage_13](../stage_13_loss_functions_batched/) then lifts the classification losses to a whole batch
> of `(B, C)` logits. Learning the softmax / cross-entropy idea on one vector first keeps it separate
> from the batching bookkeeping.

## What a loss is

Training a model means making its prediction $\hat y$ ("y-hat", the model's output) match the target $y$
(the correct answer). To do that with gradient descent you need a single number that says *how wrong the
model currently is* — a smaller number means a better model. That number is the **loss**.

Formally a loss is a function $L(\hat y, y) \to \mathbb{R}$: it takes the prediction and the target and
returns one scalar. Because the loss is built entirely from `Tensor` operations, calling
`loss.backward()` walks the graph and fills in $\partial L / \partial(\text{parameters})$ (the gradient
of the loss with respect to each parameter) for free; the optimizer then steps the parameters in the
direction that lowers $L$.

This stage builds the three losses you will use for the rest of the curriculum:

- **MSE** and **MAE** for regression (predicting real numbers).
- **Cross-Entropy** for classification (predicting a class), together with the **softmax** that turns
  raw scores into probabilities.

Everything is written as plain functions over the broadcast-capable `Tensor` from `stage_11`.

**Symbols used below:** $\hat y, y$ — prediction and target (regression). $N$ — number of elements being
averaged. For classification: $z$ — the logit vector (raw scores), $C$ — number of classes, $z_c$ — the
score of class $c$ (the index $k$ is the same kind of class index, used when summing over all classes).
$p$ — the probability vector from softmax, $p_c$ — probability of class $c$. $t$ — the true class index,
so $p_t$ is the probability the model gave the correct class. $y$ — the one-hot target vector ($y_c = 1$
for $c = t$, else $0$).

## Reductions come first

Each loss starts from a per-element error array — one error per prediction — and must collapse it to a
single scalar. That collapsing step is a **reduction**, and `stage_11`'s `Tensor` does not have one yet,
so this stage adds `sum` and `mean` (forward *and* backward) before any loss can be written.

The forward pass is just `np.sum` / `np.mean`. The interesting part is the backward pass. A reduction
maps many input elements to fewer output elements, so its gradient must do the reverse: take the
upstream gradient on the (small) output and **spread it back** onto every input element that fed into
it.

**sum.** Every input element contributes to exactly one output cell, and it does so with a coefficient
of $1$ (the output is literally the input added up). So the gradient of an input element is just the
upstream gradient of the output cell it landed in:

$$\frac{\partial L}{\partial x} = \text{(upstream grad) broadcast back to } x.\text{shape}.$$

Concretely: when `keepdims=False` the reduced axes were dropped, so first restore them as size-1 axes,
then broadcast up to the original shape.

**mean.** A mean is a sum divided by $N$, where $N$ is the number of elements that were averaged
together (the product of the reduced-axis sizes, or `x.size` when reducing everything). Dividing the
forward by a constant divides the backward by the same constant, so the mean gradient is the sum
gradient scaled by $1/N$:

$$\frac{\partial L}{\partial x} = \frac{1}{N}\,\big(\text{upstream grad expanded to } x.\text{shape}\big).$$

## Regression losses

### Mean Squared Error (MSE)

MSE penalizes the *squared* gap between prediction and target, averaged over all elements:

$$L = \frac{1}{N}\sum_i (\hat y_i - y_i)^2.$$

Differentiating the square gives a gradient that is proportional to the error itself — big errors get
big gradients, which is why MSE pulls hard on outliers:

$$\frac{\partial L}{\partial \hat y_i} = \frac{2}{N}\,(\hat y_i - y_i).$$

### Mean Absolute Error (MAE)

MAE penalizes the *absolute* gap instead of the squared one:

$$L = \frac{1}{N}\sum_i |\hat y_i - y_i|.$$

The absolute value grows linearly, so a single large error no longer dominates — MAE is the **robust**
choice when the data has outliers. Its gradient has constant magnitude $1/N$ and only the sign of the
error decides the direction:

$$\frac{\partial L}{\partial \hat y_i} = \frac{1}{N}\,\mathrm{sign}(\hat y_i - y_i).$$

At $\hat y_i = y_i$ the function has a kink (no derivative); we use the subgradient $0$ there, which is
why gradchecks for MAE must stay away from the exact kink.

## Classification: softmax + cross-entropy

A classifier outputs a vector of raw scores called **logits**, $z \in \mathbb{R}^{C}$, one per class.
Two steps turn that into a loss: softmax converts the logits to a probability distribution, then
cross-entropy measures how far that distribution is from the true label.

### Softmax

Softmax exponentiates each logit and normalizes so the result is a valid probability distribution
(non-negative, sums to 1):

$$p_c = \frac{e^{z_c}}{\sum_k e^{z_k}}.$$

Computed naively this overflows: a logit of `1000` makes $e^{z_c}$ infinite. The fix uses the fact that
shifting every logit by the same constant leaves softmax unchanged (the shift cancels top and bottom).
Subtract the largest logit $m = \max_k z_k$, so the biggest exponent is $e^{0}=1$ and nothing overflows:

$$p_c = \frac{e^{z_c - m}}{\sum_k e^{z_k - m}}.$$

The same shift gives a stable way to compute the log of the normalizer $\sum_k e^{z_k}$ — the
**log-sum-exp** identity (written $\mathrm{LSE}(z)$):

$$\mathrm{LSE}(z) = \log \sum_k e^{z_k} = m + \log \sum_k e^{z_k - m}.$$

The max $m$ is treated as a **constant** during backprop (no gradient flows through the shift); it only
exists for numerical stability.

### Cross-Entropy

Cross-entropy measures the loss of predicting distribution $p$ when the true label is the one-hot vector
$y$ (with a single $1$ at the true class $t$):

$$L = -\sum_c y_c \log p_c = -\log p_t.$$

So the loss is just the negative log-probability the model assigned to the correct class — confident and
correct gives loss near $0$, confident and wrong gives a large loss.

Rather than compute $p$ and then take its log (which reintroduces the overflow problem), expand directly
with log-sum-exp. Using $\log p_t = z_t - \log\sum_k e^{z_k}$:

$$L = \mathrm{LSE}(z) - z_t = \Big(m + \log \sum_k e^{z_k - m}\Big) - z_t.$$

The reward for going through softmax + log is the famously clean gradient — the predicted probabilities
minus the true label:

$$\frac{\partial L}{\partial z_c} = p_c - y_c.$$

Working out *why* the softmax Jacobian and the $\log$ combine into this simple $p - y$ is the whole point
of the stage — build the loss from `log_softmax` and let `backward()` reproduce it, do not hard-code it.
(In [stage_13](../stage_13_loss_functions_batched/) the same gradient appears as $(p - y)/B$ once you
average over a batch of $B$ examples.)

## Watch

- [Neural Networks Part 5: ArgMax and SoftMax](https://www.youtube.com/watch?v=KpKog-L9veg) — StatQuest: what softmax computes and why.
- [The spelled-out intro to neural networks (micrograd)](https://www.youtube.com/watch?v=VMj-3S1tku0) — Karpathy; the cross-entropy + softmax loss segment shows the `p - y` gradient in code.

## Cumulative build

Import the `Tensor` from `stage_11` via `dlfs.stage_import` and **subclass** it to add the `sum` / `mean`
reduction ops (with correct backward). On top of those reductions, add the single-example losses
`mse_loss`, `mae_loss`, and `cross_entropy_loss` (plus `softmax` / `log_softmax` via log-sum-exp). How
`cross_entropy_loss` turns one integer label into something it can multiply against the `(C,)` log-probs
(a one-hot vector, an index lookup, or whatever you choose) is left to you. This stage is where reductions
enter the engine — not just the losses.

## Exercise

In `code.py`, implement the following as **functions** that take and return `Tensor`s (import
`stage_11`'s `Tensor` via `dlfs.stage_import`; do not reimplement it). Build every loss out of `Tensor`
ops so gradients flow through `Tensor.backward()` — never hand-write a `.grad`. Allowed tools: `numpy`
(forward array construction only), the Python stdlib, and the `Tensor` engine. No PyTorch / autograd.

- `Tensor.sum(axis=None, keepdims=False) -> Tensor` and `Tensor.mean(axis=None, keepdims=False) -> Tensor`:
  add these reduction methods to the `Tensor` (subclass the imported base). Forward is `np.sum` /
  `np.mean`; backward expands the upstream grad back to the input shape over the reduced axes (mean also
  divides by the reduced count $N$).
- `mse_loss(pred, target) -> Tensor`: mean of $(\hat y - y)^2$ over all elements; scalar `Tensor`.
  `pred` / `target` are `Tensor` / array-like of equal shape (e.g. `(N,)`).
- `mae_loss(pred, target) -> Tensor`: mean of $|\hat y - y|$; scalar `Tensor`. Build `abs` from `Tensor`
  ops (e.g. $|d| = \mathrm{relu}(d) + \mathrm{relu}(-d)$) — no autodiff helper.
- `log_softmax(logits) -> Tensor`: a `(C,)` logit vector in, `(C,)` of $\log p$ out, via stable
  log-sum-exp (subtract the max logit as a constant — no grad through the shift). Equals
  `logits - logsumexp(logits)`.
- `softmax(logits) -> Tensor`: a `(C,)` probability vector; entries sum to 1. May be `exp(log_softmax(...))`.
- `cross_entropy_loss(logits, target) -> Tensor`: `logits` `(C,)`; `target` either a single integer class
  index `t` **or** a `(C,)` one-hot / soft-label vector. Return the scalar $-\log p_t$ (equivalently
  $-\sum_c y_c \log p_c$). Build it from `log_softmax` so the gradient is `p - y`.

Inputs are coerced to `Tensor` when needed.

## Done when

- `pytest stage_12_loss_functions/test.py` passes.
- `Tensor.sum` / `Tensor.mean` work with and without `axis` and with `keepdims`; their backward
  gradchecks against central differences (e.g. `f = (x*x).sum()` has analytic grad `2x`; `mean` grad is
  all `1/N`).
- Forward values match NumPy reference losses; the softmax vector sums to 1; cross-entropy is stable for
  logits like `[1000, 1001, 1002]` (no `inf` / `nan`).
- Central-difference gradcheck of each loss w.r.t. its prediction / logits matches the analytic
  `Tensor.grad` within `atol ~ 1e-6` (MAE checked away from the kink).
- Cross-entropy gradient w.r.t. logits equals `softmax(logits) - onehot(target)`.
