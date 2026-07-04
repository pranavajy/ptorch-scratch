# Stage 29: Multi-Head Attention

**Context** — A single attention head (`stage_28`) computes one set of weights over the sequence — one "way of looking" at it. Real Transformers run **h heads in parallel**, each attending over its own low-dimensional slice of the model width, then merge them with a learned output projection. This stage builds `MultiHeadAttention` (alias `MHA`) by *composing* `h` of your stage_28 `SelfAttention` heads — no attention math is rewritten, and every gradient still arrives via `Tensor.backward()`.

**Background** — Why more than one head? A single softmax produces a single convex mixture per query: one head can attend to *one* pattern at a time (e.g. the previous token *or* a matching bracket, not both). Splitting the model width $d_\text{model}$ into $h$ subspaces of width $d_k = d_\text{model}/h$ lets each head learn its own projections $W_q^{(i)}, W_k^{(i)}, W_v^{(i)} \in \mathbb{R}^{d_\text{model}\times d_k}$ and its own attention pattern; the concatenation carries all $h$ patterns forward, and $W_o \in \mathbb{R}^{d_\text{model}\times d_\text{model}}$ learns how to mix them:

$$\mathrm{MHA}(x) = \big[\mathrm{head}_1(x)\,\Vert\,\cdots\,\Vert\,\mathrm{head}_h(x)\big]\,W_o, \qquad \mathrm{head}_i(x) = \mathrm{Attention}\big(xW_q^{(i)},\,xW_k^{(i)},\,xW_v^{(i)}\big).$$

Total parameter count and FLOPs are (about) the same as one full-width head — you trade one $d_\text{model}$-wide attention for $h$ cheaper $d_k$-wide ones. Each head's $1/\sqrt{d_k}$ scale now uses the *head* width $d_k$, not $d_\text{model}$.

**The concat trick (read this — it is the whole exercise)** — Your `Tensor` has no `concat` op, and unwrapping `.data` to call `np.concatenate` would cut the graph (no gradients). But concatenation along the feature axis is *linear*, so it can be written with ops you already have: give each head a constant **placement matrix** $P_i \in \mathbb{R}^{d_k \times d_\text{model}}$ that is the $d_k \times d_k$ identity sitting in block $i$ (zeros elsewhere). Then

$$\big[\,O_1 \Vert \cdots \Vert O_h\,\big] = \sum_{i=1}^{h} O_i\,P_i,$$

built entirely from `@` and `+`, so `backward()` routes each head's slice of the upstream gradient back to that head automatically ($\partial L/\partial O_i = G_\text{concat} P_i^\top$ — exactly the block-slice of $G$). The $P_i$ are constant `Tensor`s: they sit in the graph, their grads are computed and ignored.

**Watch**

- [Attention in transformers, visually explained (3Blue1Brown, ch. 6)](https://www.youtube.com/watch?v=eMlx5fFNoYc) — the multi-head section shows heads as parallel subspace lookups.
- [Let's build GPT: from scratch (Karpathy)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the multi-head block (~1:20:00) is exactly this stage: h small heads, concat, project.

**Framework chain** — Imports `Tensor` (`stage_08` engine via `stage_12`, its latest extension) and the single-head `SelfAttention` + `causal_mask` (`stage_28`); adds `MultiHeadAttention` on top. No class is redefined; every gradient flows through `Tensor.backward()`.

**Exercise** — In `code.py`, implement `MultiHeadAttention`:

- `__init__(d_model, h, causal=False, seed=None)`: raise `ValueError` unless `d_model % h == 0`; set `d_k = d_model // h`; build `h` stage_28 `SelfAttention(d_model, d_k, causal=causal, seed=...)` heads with a **distinct seed per head** (e.g. `seed + i` — identical seeds would make identical heads), and `W_o` of shape `(d_model, d_model)` with small init scaled `1/sqrt(d_model)`; set `last_attn = None`.
- `forward(x)`: `(T, d_model) -> (T, d_model)`. Run every head on `x`, merge the `(T, d_k)` outputs into `(T, d_model)` **with the placement-matrix sum above** (Tensor ops only — no `.data` detour), record each head's `(T, T)` attention matrix in `self.last_attn`, return `concat @ W_o`.
- `__call__(x)`: alias for `forward`.
- `attention_weights(x)`: run `forward`, return each head's attention as a list of `(T, T)` NumPy arrays.
- `parameters()`: every head's `W_q, W_k, W_v` (head 0 first), then `W_o` — stable order, matching `3*h + 1` tensors.
- `zero_grad()`: zero every parameter's `.grad`.
- Allowed tools: NumPy for the constant $P_i$ blocks and inits; all differentiable math through `Tensor` ops. **No** hand-written gradients — that is stage_28's engine's job.

**Done when** `pytest stage_29_multi_head_attention/test.py` passes: forward shape `(T, d_model)`; `h=1` reduces to one `SelfAttention` head followed by `W_o`; `d_model % h != 0` raises `ValueError`; and a central-difference gradcheck of a scalar loss w.r.t. `W_o`, each head's `W_q/W_k/W_v`, and the input `x` matches the autodiff gradients within tolerance.
