# Deep Learning: Exam Notes & Formula Sheet (Lectures 2–5)

Prepared for B Nadar · Source: CS6910 lecture slides (Modules 2.1–5.8), history and biology left out.

---

# PART A: FORMULA SHEET

## A1. The three neuron models

| Model | Rule | Key points |
|---|---|---|
| McCulloch-Pitts (MP) | $g(x)=\sum_i x_i$, $x_i\in\{0,1\}$. $y=1$ if $g(x)\geq\theta$, else $0$.
If any inhibitory input is $1$, $y=0$. | Boolean inputs, fixed weights, hand-set $\theta$ |
| Perceptron | $y=1$ if $\sum_{i=0}^{n}w_ix_i\geq0$, else $0$, with $x_0=1$, $w_0=-\theta$. | Real inputs, learnable weights and threshold. Step function (harsh, not differentiable at the threshold) |
| Sigmoid (logistic) | $y=\frac{1}{1+\exp(-(w_0+\sum_{i=1}^{n}w_ix_i))}$. One input: $f(x)=\frac{1}{1+e^{-(wx+b)}}$, with $b=w_0$. | Smooth, continuous, differentiable. $y\in(0,1)$, read as a probability |

MP gates:

- AND: $\theta=n$
- OR: $\theta=1$
- NOR: $\theta=0$ with inhibitory inputs
- NOT: $\theta=0$ with an inhibitory input
- Tautology: $\theta=0$

MP geometry: The line

$$
\sum_i x_i-\theta=0
$$

splits the inputs. Points with $\sum_i x_i\geq\theta$ give $1$, the rest give $0$.

## A2. Perceptron Learning Algorithm (PLA)

- $P$ = inputs with label $1$, $N$ = inputs with label $0$. Initialize $w$ randomly.
- Repeat: pick a random $x\in P\cup N$.
  - If $x\in P$ and $w^Tx<0$:

  $$
  w=w+x
  $$

  - If $x\in N$ and $w^Tx\geq0$:

  $$
  w=w-x
  $$

- Stop when every point is classified correctly.
- Why it works: $\cos(\alpha_{\text{new}})\propto w^Tx+x^Tx>\cos(\alpha)$ after adding $x$. Subtracting $x$ does the reverse, which is what $N$ needs.
- Positive half-space: angle between $x$ and $w$ is below $90^\circ$. Negative half-space: above $90^\circ$.

**Convergence theorem:** If $P$ and $N$ are finite and linearly separable, PLA makes only a finite number of updates.

Proof skeleton:

1. Replace $N$ by negated points $N^{-}$, so $P'=P\cup N^{-}$, and every $p\in P'$ should satisfy $w^Tp\geq0$. Normalize $\|p\|=1$.
2. Let $w^*$ be a normalized separating solution and:

$$
\delta=\min_i(w^*\cdot p_i)
$$

3. After $k$ corrections, numerator:

$$
w^T w_{t+1}\geq w^T w_0+k\delta
$$

4. Denominator:

$$
\|w_{t+1}\|^2\leq\|w_0\|^2+k
$$

(because $w_t\cdot p_i\leq0$ when a correction happens).

5. So:

$$
\cos\beta\geq
\frac{w^*\cdot w_0+k\delta}
{\sqrt{\|w_0\|^2+k}}
$$

which grows with $k$. Since $\cos\beta\leq1$, $k$ must be bounded.

## A3. Counting and representation facts

| Fact | Value |
|---|---|
| Boolean functions of $n$ inputs | $2^{2^n}$ |
| 2-input Boolean functions | 16 total; all linearly separable except XOR and XNOR ($\neg XOR$) |
| Single perceptron or MP unit | Only linearly separable functions |
| Perceptron network, 1 hidden layer | Any Boolean function with $2^n$ hidden perceptrons + 1 output perceptron (sufficient, not necessary, and exponential in $n$) |
| Sigmoid network, 1 hidden layer, enough neurons | Approximates any continuous $f:\mathbb{R}^n\rightarrow\mathbb{R}^m$ within $\varepsilon$, $|g(x)-f(x)|<\varepsilon$ (Cybenko 1989, Hornik 1991) |

**Boolean network construction:** True = $+1$, False = $-1$.

Each hidden perceptron has bias $-n$ ($-2$ for 2 inputs, $-3$ for 3 inputs) and layer-1 weights of $\pm1$, so it fires for exactly one input pattern.

The output weights $w_1,\ldots,w_{2^n}$ are then chosen to satisfy one inequality per input row:

- $w_i\geq w_0$ for rows where the function is $1$.
- $w_i<w_0$ where it is $0$.

For XOR:

$$
w_1<w_0,\quad
w_2\geq w_0,\quad
w_3\geq w_0,\quad
w_4<w_0
$$

## A4. Sigmoid neuron: loss and gradient descent

- Sigmoid derivative:

$$
\sigma'(z)=\sigma(z)(1-\sigma(z))
$$

- Single-point loss:

$$
L(w,b)=\frac{1}{2}(f(x)-y)^2
$$

- Gradient with respect to weights:

$$
\nabla_w=(f(x)-y)\cdot f(x)\cdot(1-f(x))\cdot x
$$

- Gradient with respect to bias:

$$
\nabla_b=(f(x)-y)\cdot f(x)\cdot(1-f(x))
$$

- Several points: Sum the per-point gradients.
- Update rule:

$$
w_{t+1}=w_t-\eta\nabla w_t
$$

$$
b_{t+1}=b_t-\eta\nabla b_t
$$

- Taylor view:

$$
L(\theta+\eta u)=
L(\theta)
+\eta u^T\nabla L(\theta)
+\frac{\eta^2}{2!}u^T\nabla^2L(\theta)u+\cdots
$$

We need:

$$
u^T\nabla L(\theta)<0
$$

With $u^T\nabla L=k\cos\beta$, it is most negative at $\beta=180^\circ$, so move opposite the gradient:

$$
\theta_{\text{new}}=\theta-\eta\nabla L
$$

- Slide worked example: Data $(0.5,0.2)$ and $(2.5,0.9)$. With $f(2.5)=0.99$, $f(0.5)=0.62$:

$$
L=\frac{1}{2}\left((0.9-0.99)^2+(0.2-0.62)^2\right)=0.099
$$

The loss reaches about $0$ near $w=1.78$, $b=-2.27$.

## A5. Feedforward network ($L$ layers = $L-1$ hidden + 1 output)

- Initial input:

$$
h_0=x
$$

- Pre-activation:

$$
a_i=b_i+W_ih_{i-1}
$$

- Activation:

$$
h_i=g(a_i),\qquad i=1,\ldots,L-1
$$

- Output:

$$
\hat{y}=h_L=O(a_L)
$$

- Shapes:

$$
W_i\in\mathbb{R}^{n\times n},
\qquad b_i\in\mathbb{R}^n,
\qquad 0<i<L
$$

$$
W_L\in\mathbb{R}^{k\times n},
\qquad b_L\in\mathbb{R}^k
$$

- For $L=3$:

$$
\hat{y}=
O\left(
W_3g\left(W_2g(W_1x+b_1)+b_2\right)+b_3
\right)
$$

- Parameters:

$$
\theta=[W_1,\ldots,W_L,b_1,\ldots,b_L]
$$

Training is the same GD:

$$
\theta_{t+1}=\theta_t-\eta\nabla\theta_t
$$

## A6. Output activation and loss
Softmax: $\hat{y}\_j=\frac{\exp(a\_{L,j})}{\sum_i\exp(a_{L,i})}$
Cross-entropy: $L=-\sum_c y_c\log\hat{y}\_c=-\log\hat{y}_\ell$ ($\ell$ = true class)
| Output type | Output activation $O$ | Loss |
|---|---|---|
| Real values | Linear: $O(a_L)=W_Oa_L+b_O$ | Squared error: $\frac{1}{N}\sum_i\sum_j(\hat{y}\_{ij} - y\_{ij})^2$ |
| Probabilities ($k$ classes) | Softmax: $\hat{y}\_j=\frac{\exp(a\_{L,j})}{\sum_i\exp(a_{L,i})}$ | Cross-entropy: $L=-\sum_c y_c\log\hat{y}\_c=-\log\hat{y}_\ell$ ($\ell$ = true class) |

- A sigmoid output is wrong for real-valued targets because it is capped to $(0,1)$.
- One-hot $y$ is a probability distribution, so $\hat{y}$ must be one too, which is why softmax is used.
- $\hat{y}\_\ell$ is the predicted probability of the true class. $\log\hat{y}\_\ell$ is the log-likelihood, so minimizing $-\log\hat{y}_\ell$ means maximizing the likelihood.

## A7. Backpropagation (softmax output + cross-entropy)

| Quantity | Formula |
|---|---|
| w.r.t. $\hat{y}$ | $\frac{\partial L}{\partial \hat{y}\_i}=-\frac{\mathbf{1}(i=\ell)}{\hat{y}_\ell}$, i.e. $\nabla_{\hat{y}}L=-\frac{1}{\hat{y}\_\ell}e_\ell$ |
| w.r.t. $a_L$ | $\frac{\partial L}{\partial a_{L,i}}=-(\mathbf{1}(\ell=i)-\hat{y}_i)$, i.e. $\nabla_{a_L}L=-(e_\ell-\hat{y})=\hat{y}-e_\ell$ |
| w.r.t. $h_{k-1}$ | $\nabla_{h_{k-1}}L=W_k^T\nabla_{a_k}L$ |
| w.r.t. $a_{k-1}$ | $\nabla_{a_{k-1}}L=\nabla_{h_{k-1}}L\odot g'(a_{k-1})$ (element-wise) |
| w.r.t. $W_k$ | $\nabla_{W_k}L=\nabla_{a_k}L\cdot h_{k-1}^T$ (outer product; entry $\frac{\partial L}{\partial W_{kij}}=\frac{\partial L}{\partial a_{k,i}}\cdot h_{k-1,j}$) |
| w.r.t. $b_k$ | $\nabla_{b_k}L=\nabla_{a_k}L$ |

Chain rule over multiple paths:

$$
\frac{\partial L}{\partial h_{ij}}=
\sum_m
\left(
\frac{\partial L}{\partial a_{i+1,m}}
\right)
W_{i+1,m,j}
$$

Activation derivatives:

- Logistic:

$$
g'(z)=g(z)(1-g(z))
$$

- Tanh:

$$
g'(z)=1-g(z)^2
$$

Algorithm order:

1. Forward pass: For $k=1,\ldots,L-1$, compute $a_k,h_k$; then $a_L,\hat{y}$.
2. Set:

$$
\nabla_{a_L}L=-(e_\ell-\hat{y})
$$

3. For $k=L$ down to $1$: get $\nabla W_k$, $\nabla b_k$, then $\nabla h_{k-1}$, then $\nabla a_{k-1}$.
4. Update:

$$
\theta_{t+1}=\theta_t-\eta\nabla\theta_t
$$

## A8. Optimizers

| Method | Update | Notes |
|---|---|---|
| Vanilla (batch) GD | $w_{t+1}=w_t-\eta\nabla w_t$ | Uses the full dataset. 1 step per epoch |
| Momentum | $u_t=\beta u_{t-1}+\nabla w_t$, $w_{t+1}=w_t-\eta u_t$. Start $u_{-1}=0$, $u_0=\nabla w_0$, $0\leq\beta<1$ | Unrolled: $u_t=\sum_{\tau=0}^{t}\beta^{t-\tau}\nabla w_\tau$ (exponentially weighted history) |
| NAG | $u_t=\beta u_{t-1}+\nabla(w_t-\beta u_{t-1})$, $w_{t+1}=w_t-\eta u_t$ | Gradient taken at the look-ahead point $w_t-\beta u_{t-1}$ ('look before you leap') |
| SGD | One update per data point | $N$ steps per epoch |
| Mini-batch GD | One update per batch of $B$ points | $N/B$ steps per epoch |

When a question gives NAG with a bias too, the look-ahead applies to every parameter:

$$
w_{\text{la}}=w_t-\beta u_{t-1}
$$

$$
b_{\text{la}}=b_t-\beta u^b_{t-1}
$$

**Caution:** The lecture formula puts $\eta$ outside:

$$
w=w-\eta u
$$

The slide code uses $v=\beta v+\eta\nabla$ with $w=w-v$ ($\eta$ inside). These are not numerically identical, so follow the form stated in the question; the course formula is $\eta$ outside.

## A9. Learning-rate and momentum schedules

| Schedule | Formula |
|---|---|
| Step decay | Halve $\eta$ every 5 epochs, or halve it when validation error rises versus the previous epoch |
| Exponential decay | $\eta=\eta_0e^{-kt}$ (the slide prints it as $\eta_0^{(-kt)}$) |
| $1/t$ decay | $\eta=\frac{\eta_0}{1+kt}$ |
| Momentum schedule (Sutskever 2013) | $\beta_t=\min\left(1-2^{-1-\log_2(\lfloor t/250\rfloor+1)},\beta_{\max}\right)$, with $\beta_{\max}\in\{0.999,0.995,0.99,0.9,0\}$ |
| Initial $\eta$ search | Try $0.0001,0.001,0.01,0.1,1.0$ for a few epochs, then search finely around the best (e.g. best $0.1$, then try $0.05,0.2,0.3$) |
| Line search | Each step, try $\eta\in\{0.1,0.5,1,2,10\}$ and keep the update with the lowest loss |

---

# PART B: CONCISE NOTES

## B1. MP neuron vs perceptron vs sigmoid

- MP: Boolean inputs, no learnable weights, threshold set by hand. Perceptron: real inputs, weights and threshold learned. Both split the input space into two halves with a line (plane in 3D), so both handle only linearly separable functions.
- $w_0=-\theta$ is called the bias because it encodes the prior preference. Low threshold means accepts almost anything; high threshold means very selective.
- Perceptron problem: a sudden jump from 0 to 1 at the threshold (rating 0.49 vs 0.51 gives opposite decisions). Sigmoid fixes this with a smooth change and gives a probability.
- XOR is not linearly separable: the four inequalities contradict each other (the fourth, $w_1+w_2<-w_0$, conflicts with $w_1\geq-w_0$ and $w_2\geq-w_0$).
- Real data is rarely linearly separable. We accept that error cannot always reach 0 and aim for the minimum possible error.

## B2. Learning setup

Five components: **Data** $\{x_i,y_i\}$, **Model** (e.g. $\hat{y}=\frac{1}{1+e^{-w^Tx}}$), **Parameters** ($w,b$), **Learning algorithm** (PLA, gradient descent), **Loss** (guides learning; we minimize it).

## B3. Gradient descent essentials

- Error surface is plotted over $(w,b)$. Random or manual search is hopeless, so use the gradient.
- Move by $\eta\cdot\Delta\theta$ (small $\eta$ = conservative). The best direction is exactly opposite the gradient.
- Update size is proportional to the gradient. On gentle slopes the gradient is tiny, so progress is slow, regardless of start point.
- Contour plots: closely spaced contours mean a steep slope; widely spaced contours mean a gentle slope.
- A big $\eta$ does not fix gentle slopes: it blows up the already large gradients in steep regions.

## B4. Representation power

- Perceptron network: any Boolean function with one hidden layer of $2^n$ units (each hidden unit responsible for one input pattern, each output weight for one row of the truth table). Cost is exponential growth in $n$.
- Sigmoid network: with a very large $w$, a sigmoid becomes a step function (step at $x=-b/w$). Subtracting two steps (weights $+1$ and $-1$) gives a tower. In 2D, combine towers built along $x_1$ and $x_2$, then cut the tails. Many towers added together approximate any continuous surface, which is the intuition behind universal approximation.

## B5. Why cross-entropy with softmax

- Squared error + linear output is for real-valued targets (e.g. predicting ratings).
- Classification targets are one-hot (a probability distribution), so use softmax outputs and cross-entropy. The simple gradient $\hat{y}-e_\ell$ is a big payoff.

## B6. Backprop intuition

- Treat each layer as being asked 'how much are you responsible for the loss?'. Gradients are passed backwards: output, then hidden activations, then weights and biases.
- In a deep but thin network, the derivative is just a long chain-rule product. For wider networks, sum over all paths.

## B7. Optimizer comparison

- **Momentum** takes larger steps on gentle slopes (it remembers past gradients), but can overshoot and oscillate across a valley with U-turns. Slide example: after 50 iterations momentum reached error $0.00001$, while vanilla GD was stuck at $0.16$.
- **NAG** peeks ahead before computing the gradient, so it corrects course sooner, makes smaller oscillations and shorter U-turns, and is less likely to escape the minimum valley.
- **Full-batch GD:** exact gradient, loss decreases at each step in theory, but one update needs a pass over all data (very slow for large $N$).
- **SGD:** very fast updates but a noisy estimate from one point (like estimating $P(\text{heads})$ from one toss). No guarantee each step lowers the loss. Oscillations occur because each point pushes the parameters in its own favoured direction.
- **Mini-batch:** better estimate ($k=25$ visibly reduces oscillations; bigger $B$ means a more accurate estimate). Oscillations never vanish while the gradient is approximate.
- Stochastic and mini-batch versions of momentum and NAG exist. NAG still takes shorter U-turns than momentum, and both beat plain SGD.
- **Line search** reaches the minimum faster than fixed-$\eta$ GD but costs several loss evaluations per step. Its oscillations differ from those of momentum/NAG.

---

# PART C: EXAM TRAPS AND A WORKED TEMPLATE

## C1. Common traps

- Perceptron fires on $\geq0$ (not $>0$). MP fires on $\geq\theta$.
- Count of Boolean functions is $2^{2^n}$ (not $2^n$); hidden units needed is $2^n$.
- 2-input case: 14 of 16 functions are linearly separable.
- $\nabla_{a_L}L=\hat{y}-e_\ell$ only for softmax + cross-entropy.
- $\nabla_{W_k}L$ is an outer product $\nabla_{a_k}L\cdot h_{k-1}^T$; shape must match $W_k$. $\nabla_{h_{k-1}}$ uses $W_k^T$ (transpose).
- $\odot$ is element-wise multiplication with $g'(a)$.
- Momentum uses the gradient at $w_t$; NAG uses the gradient at $w_t-\beta u_{t-1}$.
- Steps per epoch: batch 1, SGD $N$, mini-batch $N/B$.
- Sigmoid gradient has the extra factor $f(1-f)$. Do not forget the $x$ factor for $\nabla_w$ but not for $\nabla_b$.

## C2. Worked template: one NAG step (sigmoid neuron, squared-error loss)

Given:

$$
x=0.5,\quad y=1,\quad \beta=0.9,\quad \eta=0.1
$$

$$
u_8=0.8,\quad w_9=2.5,\quad b_9=0
$$

1. Look-ahead:

$$
w_{\text{la}}=w_9-\beta u_8=2.5-0.72=1.78
$$

$$
b_{\text{la}}=0-0.72=-0.72
$$

2. Prediction:

$$
\hat{y}=\sigma(1.78\cdot0.5-0.72)=\sigma(0.17)\approx0.5424
$$

3. Gradient:

$$
\nabla=
(\hat{y}-y)\hat{y}(1-\hat{y})x
$$

$$=(-0.4576)(0.2482)(0.5)
\approx-0.0568
$$

(An answer key may show $-0.056$ by rounding.)

4. Update momentum:

$$
u_9=\beta u_8+\nabla=0.72-0.0568=0.6632
$$

5. Update weight:

$$
w_{10}=w_9-\eta u_9=2.5-0.06632
\approx2.4337
$$

**Note:** $1.78$ and $-0.72$ are the look-ahead point, not $w_{10}$ and $b_{10}$.

## C3. Quick self-check list

- Write the PLA update rule and say why it converges.
- Build XOR with the 4-hidden-unit network and state the output inequalities.
- Derive $\nabla_w$ for a sigmoid neuron with squared error.
- Derive $\frac{\partial L}{\partial a_{L,i}}=\hat{y}_i-\mathbf{1}(i=\ell)$.
- Write the full backprop loop with shapes.
- Contrast momentum, NAG, SGD, mini-batch, in both update rule and behaviour.
