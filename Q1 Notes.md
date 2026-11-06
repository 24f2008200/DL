# Deep Learning: Exam Notes & Formula Sheet (Lectures 2–5)

Prepared for B Nadar · Source: CS6910 lecture slides (Modules 2.1–5.8), history and biology left out. Formulas are in `code style` so subscripts stay readable.

---

## PART A: FORMULA SHEET

### A1. The three neuron models

| Model | Rule | Key points |
| --- | --- | --- |
| McCulloch-Pitts (MP) | `g(x) = Σ x_i`, `x_i ∈ {0,1}`. `y = 1` if `g(x) ≥ θ`, else `0`. If any inhibitory input is 1, `y = 0` | Boolean inputs, fixed weights, hand-set θ |
| Perceptron | `y = 1` if `Σ_{i=0..n} w_i x_i ≥ 0`, else `0`, with `x_0 = 1`, `w_0 = -θ` | Real inputs, learnable weights and threshold. Step function (harsh, not differentiable at the threshold) |
| Sigmoid (logistic) | `y = 1 / (1 + exp(-(w_0 + Σ_{i=1..n} w_i x_i)))`. One input: `f(x) = 1 / (1 + e^-(wx+b))`, with `b = w_0` | Smooth, continuous, differentiable. `y ∈ (0,1)`, read as a probability |

MP gates: AND `θ = n`; OR `θ = 1`; NOR `θ = 0` with inhibitory inputs; NOT `θ = 0` with an inhibitory input; Tautology `θ = 0`.

MP geometry: the line `Σ x_i − θ = 0` splits the inputs. Points with `Σ x_i ≥ θ` give 1, the rest give 0.

### A2. Perceptron Learning Algorithm (PLA)

- P = inputs with label 1, N = inputs with label 0. Initialize `w` randomly.
- Repeat: pick a random `x ∈ P ∪ N`.
  - If `x ∈ P` and `wᵀx < 0`: `w = w + x`
  - If `x ∈ N` and `wᵀx ≥ 0`: `w = w − x`
- Stop when every point is classified correctly.
- Why it works: `cos(α_new) ∝ wᵀx + xᵀx > cos(α)` after adding x. Subtracting x does the reverse, which is what N needs.
- Positive half-space: angle between `x` and `w` is below 90°. Negative half-space: above 90°.

**Convergence theorem:** if P and N are finite and linearly separable, PLA makes only a finite number of updates.

Proof skeleton:

1. Replace N by negated points N⁻, so `P' = P ∪ N⁻`, and every `p ∈ P'` should satisfy `wᵀp ≥ 0`. Normalize `‖p‖ = 1`.
2. Let `w*` be a normalized separating solution and `δ = min_i (w*·p_i)`.
3. After k corrections: numerator `w*·w_{t+1} ≥ w*·w_0 + kδ`.
4. Denominator: `‖w_{t+1}‖² ≤ ‖w_0‖² + k` (because `w_t·p_i ≤ 0` when a correction happens).
5. So `cos β ≥ (w*·w_0 + kδ) / sqrt(‖w_0‖² + k)`, which grows with k. Since `cos β ≤ 1`, k must be bounded.

### A3. Counting and representation facts

| Fact | Value |
| --- | --- |
| Boolean functions of n inputs | `2^(2^n)` |
| 2-input boolean functions | 16 total; all linearly separable except XOR and XNOR (!XOR) |
| Single perceptron or MP unit | Only linearly separable functions |
| Perceptron network, 1 hidden layer | Any boolean function with `2^n` hidden perceptrons + 1 output perceptron (sufficient, not necessary, and exponential in n) |
| Sigmoid network, 1 hidden layer, enough neurons | Approximates any continuous `f: R^n → R^m` within ε, `abs(g(x) − f(x)) < ε` (Cybenko 1989, Hornik 1991) |

**Boolean network construction:** True = +1, False = −1. Each hidden perceptron has bias `−n` (−2 for 2 inputs, −3 for 3 inputs) and layer-1 weights of ±1, so it fires for exactly one input pattern. The output weights `w_1..w_{2^n}` are then chosen to satisfy one inequality per input row (`w_i ≥ w_0` for rows where the function is 1, `w_i < w_0` where it is 0). For XOR: `w_1 < w_0`, `w_2 ≥ w_0`, `w_3 ≥ w_0`, `w_4 < w_0`.

### A4. Sigmoid neuron: loss and gradient descent

- Sigmoid derivative: `σ'(z) = σ(z)(1 − σ(z))`
- Single-point loss: `L(w,b) = ½ (f(x) − y)²`
- `∇w = (f(x) − y) · f(x) · (1 − f(x)) · x`
- `∇b = (f(x) − y) · f(x) · (1 − f(x))`
- Several points: sum the per-point gradients.
- Update rule: `w_{t+1} = w_t − η ∇w_t`, `b_{t+1} = b_t − η ∇b_t`
- Taylor view: `L(θ + ηu) = L(θ) + η uᵀ∇L(θ) + (η²/2!) uᵀ∇²L(θ)u + ...`. We need `uᵀ∇L(θ) < 0`. With `uᵀ∇L = k cos β`, it is most negative at `β = 180°`, so move opposite the gradient: `θ_new = θ − η ∇L`.
- Slide worked example: data (0.5, 0.2) and (2.5, 0.9). With `f(2.5) = 0.99`, `f(0.5) = 0.62`: `L = ½((0.9−0.99)² + (0.2−0.62)²) = 0.099`. The loss reaches about 0 near `w = 1.78`, `b = −2.27`.

### A5. Feedforward network (L layers = L−1 hidden + 1 output)

- `h_0 = x`
- Pre-activation: `a_i = b_i + W_i h_{i-1}`
- Activation: `h_i = g(a_i)`, for `i = 1..L−1`
- Output: `ŷ = h_L = O(a_L)`
- Shapes: `W_i ∈ R^(n×n)`, `b_i ∈ R^n` for `0 < i < L`; `W_L ∈ R^(k×n)`, `b_L ∈ R^k`
- For L = 3: `ŷ = O(W_3 g(W_2 g(W_1 x + b_1) + b_2) + b_3)`
- Parameters: `θ = [W_1..W_L, b_1..b_L]`. Training is the same GD: `θ_{t+1} = θ_t − η ∇θ_t`

### A6. Output activation and loss

| Output type | Output activation O | Loss |
| --- | --- | --- |
| Real values | Linear: `O(a_L) = W_O a_L + b_O` | Squared error: `(1/N) Σ_i Σ_j (ŷ_ij − y_ij)²` |
| Probabilities (k classes) | Softmax: `ŷ_j = exp(a_{L,j}) / Σ_i exp(a_{L,i})` | Cross-entropy: `L = −Σ_c y_c log ŷ_c = −log ŷ_ℓ` (ℓ = true class) |

- A sigmoid output is wrong for real-valued targets because it is capped to (0,1).
- One-hot `y` is a probability distribution, so `ŷ` must be one too, which is why softmax is used.
- `ŷ_ℓ` is the predicted probability of the true class. `log ŷ_ℓ` is the log-likelihood, so minimizing `−log ŷ_ℓ` means maximizing the likelihood.

### A7. Backpropagation (softmax output + cross-entropy)

| Quantity | Formula |
| --- | --- |
| w.r.t. ŷ | `∂L/∂ŷ_i = −1(i=ℓ) / ŷ_ℓ`, i.e. `∇ŷ L = −(1/ŷ_ℓ) e_ℓ` |
| w.r.t. a\_L | `∂L/∂a_{L,i} = −(1(ℓ=i) − ŷ_i)`, i.e. `∇a_L L = −(e_ℓ − ŷ) = ŷ − e_ℓ` |
| w.r.t. h\_{k-1} | `∇h_{k-1} L = W_kᵀ ∇a_k L` |
| w.r.t. a\_{k-1} | `∇a_{k-1} L = ∇h_{k-1} L ⊙ g'(a_{k-1})` (element-wise) |
| w.r.t. W\_k | `∇W_k L = ∇a_k L · h_{k-1}ᵀ` (outer product; entry `∂L/∂W_kij = ∂L/∂a_ki · h_{k-1,j}`) |
| w.r.t. b\_k | `∇b_k L = ∇a_k L` |

Chain rule over multiple paths: `∂L/∂h_ij = Σ_m (∂L/∂a_{i+1,m}) · W_{i+1,m,j}`.

Activation derivatives: logistic `g'(z) = g(z)(1 − g(z))`; tanh `g'(z) = 1 − g(z)²`.

Algorithm order:

1. Forward pass: for `k = 1..L−1` compute `a_k`, `h_k`; then `a_L`, `ŷ`.
2. Set `∇a_L L = −(e_ℓ − ŷ)`.
3. For `k = L` down to 1: get `∇W_k`, `∇b_k`, then `∇h_{k-1}`, then `∇a_{k-1}`.
4. Update `θ_{t+1} = θ_t − η ∇θ_t`.

### A8. Optimizers

| Method | Update | Notes |
| --- | --- | --- |
| Vanilla (batch) GD | `w_{t+1} = w_t − η ∇w_t` | Uses the full dataset. 1 step per epoch |
| Momentum | `u_t = β u_{t-1} + ∇w_t`, `w_{t+1} = w_t − η u_t`. Start `u_{-1} = 0`, `u_0 = ∇w_0`, `0 ≤ β < 1` | Unrolled: `u_t = Σ_{τ=0..t} β^(t−τ) ∇w_τ` (exponentially weighted history) |
| NAG | `u_t = β u_{t-1} + ∇(w_t − β u_{t-1})`, `w_{t+1} = w_t − η u_t` | Gradient taken at the look-ahead point `w_t − β u_{t-1}` ('look before you leap') |
| SGD | One update per data point | N steps per epoch |
| Mini-batch GD | One update per batch of B points | N/B steps per epoch |

When a question gives NAG with a bias too, the look-ahead applies to every parameter: `w_la = w_t − β u_{t-1}`, `b_la = b_t − β u^b_{t-1}`.

Caution: the lecture formula puts η outside (`w = w − η u`). The slide code uses `v = β v + η ∇` with `w = w − v` (η inside). These are not numerically identical, so follow the form stated in the question; the course formula is η outside.

### A9. Learning-rate and momentum schedules

| Schedule | Formula |
| --- | --- |
| Step decay | Halve η every 5 epochs, or halve it when validation error rises versus the previous epoch |
| Exponential decay | `η = η_0 e^(−kt)` (the slide prints it as `η_0^(−kt)`) |
| 1/t decay | `η = η_0 / (1 + kt)` |
| Momentum schedule (Sutskever 2013) | `β_t = min(1 − 2^(−1 − log2(floor(t/250) + 1)), β_max)`, with `β_max ∈ {0.999, 0.995, 0.99, 0.9, 0}` |
| Initial η search | Try 0.0001, 0.001, 0.01, 0.1, 1.0 for a few epochs, then search finely around the best (e.g. best 0.1, then try 0.05, 0.2, 0.3) |
| Line search | Each step, try `η ∈ {0.1, 0.5, 1, 2, 10}` and keep the update with the lowest loss |

---

## PART B: CONCISE NOTES

### B1. MP neuron vs perceptron vs sigmoid

- MP: boolean inputs, no learnable weights, threshold set by hand. Perceptron: real inputs, weights and threshold learned. Both split the input space into two halves with a line (plane in 3D), so both handle only linearly separable functions.
- `w_0 = −θ` is called the bias because it encodes the prior preference. Low threshold means accepts almost anything; high threshold means very selective.
- Perceptron problem: a sudden jump from 0 to 1 at the threshold (rating 0.49 vs 0.51 gives opposite decisions). Sigmoid fixes this with a smooth change and gives a probability.
- XOR is not linearly separable: the four inequalities contradict each other (the fourth, `w_1 + w_2 < −w_0`, conflicts with `w_1 ≥ −w_0` and `w_2 ≥ −w_0`).
- Real data is rarely linearly separable. We accept that error cannot always reach 0 and aim for the minimum possible error.

### B2. Learning setup

Five components: **Data** `{x_i, y_i}`, **Model** (e.g. `ŷ = 1/(1+e^(−wᵀx))`), **Parameters** (w, b), **Learning algorithm** (PLA, gradient descent), **Loss** (guides learning; we minimize it).

### B3. Gradient descent essentials

- Error surface is plotted over (w, b). Random or manual search is hopeless, so use the gradient.
- Move by `η·Δθ` (small η = conservative). The best direction is exactly opposite the gradient.
- Update size is proportional to the gradient. On gentle slopes the gradient is tiny, so progress is slow, regardless of start point.
- Contour plots: closely spaced contours mean a steep slope; widely spaced contours mean a gentle slope.
- A big η does not fix gentle slopes: it blows up the already large gradients in steep regions.

### B4. Representation power

- Perceptron network: any boolean function with one hidden layer of `2^n` units (each hidden unit responsible for one input pattern, each output weight for one row of the truth table). Cost is exponential growth in n.
- Sigmoid network: with a very large `w`, a sigmoid becomes a step function (step at `x = −b/w`). Subtracting two steps (weights +1 and −1) gives a tower. In 2D, combine towers built along x₁ and x₂, then cut the tails. Many towers added together approximate any continuous surface, which is the intuition behind universal approximation.

### B5. Why cross-entropy with softmax

- Squared error + linear output is for real-valued targets (e.g. predicting ratings).
- Classification targets are one-hot (a probability distribution), so use softmax outputs and cross-entropy. The simple gradient `ŷ − e_ℓ` is a big payoff.

### B6. Backprop intuition

- Treat each layer as being asked 'how much are you responsible for the loss?'. Gradients are passed backwards: output, then hidden activations, then weights and biases.
- In a deep but thin network, the derivative is just a long chain-rule product. For wider networks, sum over all paths.

### B7. Optimizer comparison

- **Momentum** takes larger steps on gentle slopes (it remembers past gradients), but can overshoot and oscillate across a valley with U-turns. Slide example: after 50 iterations momentum reached error 0.00001, while vanilla GD was stuck at 0.16.
- **NAG** peeks ahead before computing the gradient, so it corrects course sooner, makes smaller oscillations and shorter U-turns, and is less likely to escape the minimum valley.
- **Full-batch GD:** exact gradient, loss decreases at each step in theory, but one update needs a pass over all data (very slow for large N).
- **SGD:** very fast updates but a noisy estimate from one point (like estimating P(heads) from one toss). No guarantee each step lowers the loss. Oscillations occur because each point pushes the parameters in its own favoured direction.
- **Mini-batch:** better estimate (k = 25 visibly reduces oscillations; bigger B means a more accurate estimate). Oscillations never vanish while the gradient is approximate.
- Stochastic and mini-batch versions of momentum and NAG exist. NAG still takes shorter U-turns than momentum, and both beat plain SGD.
- **Line search** reaches the minimum faster than fixed-η GD but costs several loss evaluations per step. Its oscillations differ from those of momentum/NAG.

---

## PART C: EXAM TRAPS AND A WORKED TEMPLATE

### C1. Common traps

- Perceptron fires on `≥ 0` (not `> 0`). MP fires on `≥ θ`.
- Count of boolean functions is `2^(2^n)` (not `2^n`); hidden units needed is `2^n`.
- 2-input case: 14 of 16 functions are linearly separable.
- `∇a_L L = ŷ − e_ℓ` only for softmax + cross-entropy.
- `∇W_k L` is an outer product `∇a_k L · h_{k-1}ᵀ`; shape must match `W_k`. `∇h_{k-1}` uses `W_kᵀ` (transpose).
- `⊙` is element-wise multiplication with `g'(a)`.
- Momentum uses the gradient at `w_t`; NAG uses the gradient at `w_t − β u_{t-1}`.
- Steps per epoch: batch 1, SGD N, mini-batch N/B.
- Sigmoid gradient has the extra factor `f(1 − f)`. Do not forget the `x` factor for `∇w` but not for `∇b`.

### C2. Worked template: one NAG step (sigmoid neuron, squared-error loss)

Given `x = 0.5`, `y = 1`, `β = 0.9`, `η = 0.1`, `u_8 = 0.8`, `w_9 = 2.5`, `b_9 = 0`:

1. Look-ahead: `w_la = w_9 − β u_8 = 2.5 − 0.72 = 1.78`; `b_la = 0 − 0.72 = −0.72`
2. `ŷ = σ(1.78·0.5 − 0.72) = σ(0.17) ≈ 0.5424`
3. `∇ = (ŷ − y) ŷ (1 − ŷ) x = (−0.4576)(0.2482)(0.5) ≈ −0.0568` (an answer key may show −0.056 by rounding)
4. `u_9 = β u_8 + ∇ = 0.72 − 0.0568 = 0.6632`
5. `w_10 = w_9 − η u_9 = 2.5 − 0.06632 ≈ 2.4337`

Note: 1.78 and −0.72 are the look-ahead point, not `w_10` and `b_10`.

### C3. Quick self-check list

- Write the PLA update rule and say why it converges.
- Build XOR with the 4-hidden-unit network and state the output inequalities.
- Derive `∇w` for a sigmoid neuron with squared error.
- Derive `∂L/∂a_{L,i} = ŷ_i − 1(i=ℓ)`.
- Write the full backprop loop with shapes.
- Contrast momentum, NAG, SGD, mini-batch, in both update rule and behaviour.
