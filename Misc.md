I derived it by turning the momentum update into a recurrence on the error and then asking what kind of sequence it produces.

**Step 1: shift the problem so the minimum is at 0.**
Let e_t = w_t − 5. Then L = e², and the gradient is 2e_t.

**Step 2: write the update (your course's convention).**
- u_t = β·u_{t−1} + 2e_{t−1}
- e_t = e_{t−1} − η·u_t

**Step 3: eliminate u.**
From the second line, e_t − e_{t−1} = −η·u_t. One step earlier, e_{t−1} − e_{t−2} = −η·u_{t−1}, so u_{t−1} = −(e_{t−1} − e_{t−2})/η.

Substitute u_t = β·u_{t−1} + 2e_{t−1} into e_t − e_{t−1} = −η·u_t:

e_t − e_{t−1} = β(e_{t−1} − e_{t−2}) − 2η·e_{t−1}

Rearranging:

**e_t = (1 + β − 2η)·e_{t−1} − β·e_{t−2}**

With η = 0.04, 2η = 0.08.

**Step 4: get the characteristic equation.**
For a linear recurrence, try e_t = λ^t. Plugging in and dividing by λ^(t−2) gives

λ² = (1 + β − 0.08)·λ − β, which is λ² − (1 + β − 0.08)λ + β = 0.

**Why it tells you the behavior:**
- The two roots λ₁, λ₂ control how e_t evolves, as e_t = c₁λ₁^t + c₂λ₂^t.
- If the roots are **real**, the error decays smoothly with no oscillation (β = 0 and β = 0.5).
- If they are **complex**, the error oscillates. The product of the roots equals β, so both have magnitude √β. That means the oscillation shrinks by a factor of √β per step: √0.9 ≈ 0.949 and √0.99 ≈ 0.995. This is why β = 0.99 barely damps.
- The discriminant (1 + β − 0.08)² − 4β decides real versus complex. It is positive for β = 0 and 0.5, and negative for β = 0.9 and 0.99.

For β = 0 the recurrence reduces to e_t = 0.92·e_{t−1}, which is plain gradient descent.


Deep Learning: Exam Notes & Formula Sheet (Lectures 2–5)
Prepared for B Nadar · Source: CS6910 lecture slides (Modules 2.1–5.8), history and biology left out.
PART A: FORMULA SHEET
A1. Neuron models
McCulloch-Pitts (MP): boolean inputs, fixed weights, hand-set threshold θ. If any inhibitory input is 1, the output is 0.
g(\mathbf{x})=\sum_{i=1}^{n}x_i,\qquad x_i\in\{0,1\},\qquad y=\begin{cases}1 & g(\mathbf{x})\ge\theta\\ 0 & g(\mathbf{x})<\theta\end{cases}
Gates: AND has θ = n; OR has θ = 1; NOR has θ = 0 with inhibitory inputs; NOT has θ = 0 with an inhibitory input; a tautology has θ = 0. The decision line is the sum of the inputs equal to θ.
Perceptron: real inputs; weights and threshold are learned. The bias is w₀ = −θ with x₀ = 1.
y=\begin{cases}1 & \sum_{i=0}^{n}w_i x_i\ge 0\\ 0 & \sum_{i=0}^{n}w_i x_i<0\end{cases},\qquad x_0=1,\quad w_0=-\theta
Sigmoid (logistic) neuron: smooth, continuous, differentiable; the output lies in (0, 1) and is read as a probability.
y=\frac{1}{1+\exp\!\left(-\left(w_0+\sum_{i=1}^{n}w_i x_i\right)\right)},\qquad f(x)=\frac{1}{1+e^{-(wx+b)}}\ \ (b=w_0)
A2. Perceptron Learning Algorithm (PLA)
P holds inputs with label 1, N holds inputs with label 0. Initialize w randomly, then repeatedly pick a random point and apply:
\begin{aligned}
&\mathbf{x}\in P \text{ and } \mathbf{w}^{T}\mathbf{x}<0 \;\Rightarrow\; \mathbf{w}\leftarrow\mathbf{w}+\mathbf{x}\\
&\mathbf{x}\in N \text{ and } \mathbf{w}^{T}\mathbf{x}\ge 0 \;\Rightarrow\; \mathbf{w}\leftarrow\mathbf{w}-\mathbf{x}
\end{aligned}
Stop when every point is classified correctly. Why it works: adding x to w raises the cosine of the angle between them.
\cos\alpha_{new}\propto(\mathbf{w}+\mathbf{x})^{T}\mathbf{x}=\mathbf{w}^{T}\mathbf{x}+\mathbf{x}^{T}\mathbf{x}>\mathbf{w}^{T}\mathbf{x}\propto\cos\alpha
Positive half-space: angle between x and w is below 90°. Negative half-space: above 90°.
Convergence theorem: if P and N are finite and linearly separable, PLA makes only a finite number of updates.
Proof skeleton:
1. Negate the points of N to get N⁻, so every p in P′ = P ∪ N⁻ should satisfy wᵀp ≥ 0. Normalize each p to length 1.
2. Let w* be a normalized separating solution, and let δ be the smallest value of w*·p over all points.
3. After k corrections the numerator grows by at least δ each time, while the squared length of w grows by at most 1 each time:
\mathbf{w}^{*}\!\cdot\mathbf{w}_{t+1}\ \ge\ \mathbf{w}^{*}\!\cdot\mathbf{w}_{0}+k\delta,\qquad \lVert\mathbf{w}_{t+1}\rVert^{2}\ \le\ \lVert\mathbf{w}_{0}\rVert^{2}+k
\cos\beta\ \ge\ \frac{\mathbf{w}^{*}\!\cdot\mathbf{w}_{0}+k\delta}{\sqrt{\lVert\mathbf{w}_{0}\rVert^{2}+k}}
4. The right side keeps growing with k, but cos β cannot exceed 1, so k is bounded.
A3. Counting and representation facts
\#\text{ boolean functions of } n \text{ inputs}=2^{2^{n}},\qquad \#\text{ hidden perceptrons that suffice}=2^{n}
• 2 inputs: 16 functions, 14 linearly separable (all except XOR and XNOR).
• A single MP unit or perceptron represents only linearly separable functions.
• A perceptron network with one hidden layer of 2ⁿ units plus one output unit represents any boolean function (sufficient, not necessary; exponential in n).
• A sigmoid network with one hidden layer and enough neurons approximates any continuous function f: Rⁿ → Rᵐ (Cybenko 1989, Hornik 1991):
\lvert g(x)-f(x)\rvert<\epsilon
Boolean network construction: True = +1, False = −1. Each hidden perceptron has bias −n (−2 for 2 inputs, −3 for 3 inputs) and layer-1 weights of ±1, so it fires for exactly one input pattern. The output weights are then set by one inequality per truth-table row (w₀ is the output threshold). For XOR:
w_1<w_0,\quad w_2\ge w_0,\quad w_3\ge w_0,\quad w_4<w_0
A4. Sigmoid neuron: loss and gradient descent
\sigma'(z)=\sigma(z)\bigl(1-\sigma(z)\bigr),\qquad L(w,b)=\tfrac12\bigl(f(x)-y\bigr)^{2}
\begin{aligned}
\nabla w &= \bigl(f(x)-y\bigr)\,f(x)\bigl(1-f(x)\bigr)\,x\\
\nabla b &= \bigl(f(x)-y\bigr)\,f(x)\bigl(1-f(x)\bigr)
\end{aligned}
For several points, sum the per-point gradients. Update rule:
w_{t+1}=w_t-\eta\,\nabla w_t,\qquad b_{t+1}=b_t-\eta\,\nabla b_t
Why the negative gradient: the Taylor expansion is
L(\theta+\eta u)=L(\theta)+\eta\,u^{T}\nabla L(\theta)+\frac{\eta^{2}}{2!}\,u^{T}\nabla^{2}L(\theta)\,u+\cdots
We need uᵀ∇L(θ) < 0. Since uᵀ∇L = k cos β, it is most negative at β = 180°, so u points opposite the gradient and θ_new = θ − η∇L.
Slide worked example (data (0.5, 0.2) and (2.5, 0.9), with f(2.5) = 0.99 and f(0.5) = 0.62):
L=\tfrac12\bigl((0.9-0.99)^{2}+(0.2-0.62)^{2}\bigr)=0.099
The loss reaches about 0 near w = 1.78, b = −2.27.
A5. Feedforward network (L layers = L−1 hidden + 1 output)
\begin{aligned}
a_i&=b_i+W_i h_{i-1}, & h_0&=x\\
h_i&=g(a_i), & i&=1,\dots,L-1\\
\hat y&=h_L=O(a_L)
\end{aligned}
W_i\in\mathbb{R}^{n\times n},\ b_i\in\mathbb{R}^{n}\ (0<i<L);\qquad W_L\in\mathbb{R}^{k\times n},\ b_L\in\mathbb{R}^{k}
For L = 3, with parameters θ = [W₁, …, W_L, b₁, …, b_L] trained by the same gradient descent:
\hat y=O\bigl(W_3\,g(W_2\,g(W_1x+b_1)+b_2)+b_3\bigr),\qquad \theta_{t+1}=\theta_t-\eta\,\nabla\theta_t
A6. Output activation and loss
Real-valued outputs use a linear output and squared error:
O(a_L)=W_O a_L+b_O,\qquad L(\theta)=\frac1N\sum_{i=1}^{N}\sum_{j=1}^{k}(\hat y_{ij}-y_{ij})^{2}
Class probabilities use softmax and cross-entropy (ℓ is the true class):
\hat y_j=\frac{e^{a_{L,j}}}{\sum_{i=1}^{k}e^{a_{L,i}}},\qquad L(\theta)=-\sum_{c=1}^{k}y_c\log\hat y_c=-\log\hat y_\ell
• A sigmoid output is wrong for real-valued targets because it is capped to (0, 1).
• A one-hot y is a probability distribution, so ŷ must be one too; that is why softmax is used.
• ŷ_ℓ is the predicted probability of the true class; log ŷ_ℓ is the log-likelihood, so minimizing −log ŷ_ℓ maximizes the likelihood.
A7. Backpropagation (softmax output with cross-entropy)
Output layer:
\begin{aligned}
\frac{\partial L}{\partial \hat y_i}&=-\frac{\mathbb{1}_{i=\ell}}{\hat y_\ell}, & \nabla_{\hat y}L&=-\frac{1}{\hat y_\ell}\,e_\ell\\
\frac{\partial L}{\partial a_{L,i}}&=-\bigl(\mathbb{1}_{\ell=i}-\hat y_i\bigr), & \nabla_{a_L}L&=-(e_\ell-\hat y)=\hat y-e_\ell
\end{aligned}
Hidden layers and parameters (⊙ is element-wise multiplication):
\begin{aligned}
\nabla_{h_{k-1}}L&=W_k^{T}\,\nabla_{a_k}L\\
\nabla_{a_{k-1}}L&=\nabla_{h_{k-1}}L\odot g'(a_{k-1})\\
\nabla_{W_k}L&=\nabla_{a_k}L\;h_{k-1}^{T}\\
\nabla_{b_k}L&=\nabla_{a_k}L
\end{aligned}
Entry-wise forms, and the chain rule over multiple paths:
\frac{\partial L}{\partial W_{k,ij}}=\frac{\partial L}{\partial a_{k,i}}\,h_{k-1,j},\qquad \frac{\partial L}{\partial h_{ij}}=\sum_{m}\frac{\partial L}{\partial a_{i+1,m}}\,W_{i+1,m,j}
Activation derivatives:
\text{logistic: } g'(z)=g(z)\bigl(1-g(z)\bigr),\qquad\qquad \tanh\text{: } g'(z)=1-g(z)^{2}
Algorithm order:
1. Forward pass: compute a_k and h_k for k = 1 to L−1, then a_L and ŷ.
2. Set ∇ at a_L to −(e_ℓ − ŷ).
3. For k = L down to 1: get the gradients for W_k and b_k, then for h_{k−1}, then for a_{k−1}.
4. Update θ_{t+1} = θ_t − η∇θ_t.
A8. Optimizers
Vanilla (batch) gradient descent uses the full dataset, one step per epoch:
w_{t+1}=w_t-\eta\,\nabla w_t
Momentum (an exponentially weighted history of gradients):
u_t=\beta u_{t-1}+\nabla w_t,\qquad w_{t+1}=w_t-\eta\,u_t,\qquad u_{-1}=0,\ \ 0\le\beta<1
u_t=\sum_{\tau=0}^{t}\beta^{\,t-\tau}\,\nabla w_\tau
Nesterov Accelerated Gradient (NAG) takes the gradient at the look-ahead point w_t − βu_{t−1}:
u_t=\beta u_{t-1}+\nabla\bigl(w_t-\beta u_{t-1}\bigr),\qquad w_{t+1}=w_t-\eta\,u_t
When a bias is present, the look-ahead applies to every parameter (each with its own u).
Caution: the lecture formula puts η outside, as above. The slide code instead uses v = βv + η∇ and w = w − v (η inside). These are not numerically identical, so follow the form stated in the question; the course formula is η outside.
Steps per epoch: batch GD takes 1, SGD takes N, mini-batch GD takes N/B (B is the batch size).
A9. Learning-rate and momentum schedules
\text{exponential: }\eta=\eta_0 e^{-kt},\qquad\qquad \text{1/t: }\eta=\frac{\eta_0}{1+kt}
The slide prints exponential decay as η₀ raised to −kt, which looks like a typo for the form above.
\beta_t=\min\!\Bigl(1-2^{-1-\log_2\left(\lfloor t/250\rfloor+1\right)},\ \beta_{\max}\Bigr),\qquad \beta_{\max}\in\{0.999,\,0.995,\,0.99,\,0.9,\,0\}
• Step decay: halve η every 5 epochs, or halve it when validation error rises compared with the previous epoch.
• Initial η search: try 0.0001, 0.001, 0.01, 0.1, 1.0 for a few epochs, then search finely around the best (for example, if 0.1 was best, try 0.05, 0.2, 0.3).
• Line search: each step, try η in {0.1, 0.5, 1, 2, 10} and keep the update with the lowest loss.
PART B: CONCISE NOTES
B1. MP neuron vs perceptron vs sigmoid
• MP: boolean inputs, no learnable weights, threshold set by hand. Perceptron: real inputs, weights and threshold learned. Both split the input space with a line (a plane in 3D), so both handle only linearly separable functions.
• The bias w₀ = −θ encodes the prior preference: a low threshold accepts almost anything, a high threshold is very selective.
• Perceptron problem: a sudden jump from 0 to 1 at the threshold (rating 0.49 and 0.51 give opposite decisions). The sigmoid fixes this with a smooth change and gives a probability.
• XOR is not linearly separable: the four inequalities contradict each other. The fourth, w₁ + w₂ < −w₀, conflicts with w₁ ≥ −w₀ and w₂ ≥ −w₀.
• Real data is rarely linearly separable, so we accept that error cannot always reach 0 and aim for the minimum possible error.
B2. Learning setup
Five components: Data (pairs xᵢ, yᵢ), Model (for example ŷ = 1 / (1 + e^(−wᵀx))), Parameters (w, b), Learning algorithm (PLA, gradient descent) and Loss (guides learning; we minimize it).
B3. Gradient descent essentials
• The error surface is plotted over (w, b). Random or manual search is hopeless, so use the gradient.
• Move by η·Δθ (small η is conservative). The best direction is exactly opposite the gradient.
• Update size is proportional to the gradient, so on gentle slopes progress is slow regardless of the starting point.
• Contour plots: closely spaced contours mean a steep slope; widely spaced contours mean a gentle slope.
• A big η does not fix gentle slopes: it blows up the already large gradients in steep regions.
B4. Representation power
• Perceptron network: any boolean function with one hidden layer of 2ⁿ units. Each hidden unit is responsible for one input pattern and each output weight for one truth-table row. The cost is exponential growth in n.
• Sigmoid network: with a very large w, a sigmoid becomes a step function (the step sits at x = −b/w). Subtracting two steps (weights +1 and −1) gives a tower. In 2D, combine towers built along x₁ and x₂, then cut the tails. Adding many towers approximates any continuous surface, which is the intuition behind universal approximation.
B5. Why cross-entropy with softmax
• Squared error with a linear output suits real-valued targets (for example predicting ratings).
• Classification targets are one-hot (a probability distribution), so use softmax outputs with cross-entropy. The simple output gradient ŷ − e_ℓ is a big payoff.
B6. Backprop intuition
• Each layer is asked how much it is responsible for the loss. Gradients are passed backwards: output, then hidden activations, then weights and biases.
• In a deep but thin network the derivative is one long chain-rule product. In wider networks, sum over all paths.
B7. Optimizer comparison
• Momentum takes larger steps on gentle slopes because it remembers past gradients, but it can overshoot and oscillate across a valley with U-turns. Slide example: after 50 iterations momentum reached error 0.00001 while vanilla GD was stuck at 0.16.
• NAG peeks ahead before computing the gradient, so it corrects course sooner, makes smaller oscillations and shorter U-turns, and is less likely to escape the minimum valley.
• Full-batch GD: exact gradient, and in theory the loss decreases at every step, but one update needs a pass over all data (very slow for large N).
• SGD: fast updates but a noisy estimate from one point (like estimating P(heads) from one toss). There is no guarantee each step lowers the loss. Oscillations occur because each point pushes the parameters in its own favoured direction.
• Mini-batch: a better estimate (k = 25 visibly reduces oscillations; bigger B means a more accurate estimate). Oscillations never vanish while the gradient is approximate.
• Stochastic and mini-batch versions of momentum and NAG exist. NAG still takes shorter U-turns than momentum, and both beat plain SGD.
• Line search reaches the minimum faster than fixed-η GD but costs several loss evaluations per step. Its oscillations differ from those of momentum and NAG.
PART C: EXAM TRAPS AND A WORKED TEMPLATE
C1. Common traps
• The perceptron fires on ≥ 0 (not > 0). The MP neuron fires on sum ≥ θ.
• The count of boolean functions is 2 to the power 2ⁿ (not 2ⁿ); the hidden units needed number 2ⁿ.
• 2-input case: 14 of 16 functions are linearly separable.
• The output gradient ŷ − e_ℓ holds only for softmax with cross-entropy.
• The gradient for W_k is an outer product of the gradient at a_k with h_{k−1} transposed; its shape must match W_k. The gradient for h_{k−1} uses W_k transposed.
• Momentum uses the gradient at w_t; NAG uses the gradient at w_t − βu_{t−1}.
• Steps per epoch: batch 1, SGD N, mini-batch N/B.
• The sigmoid gradient has the extra factor f(1 − f). The gradient for w has a factor x; the gradient for b does not.
C2. Worked template: one NAG step (sigmoid neuron, squared-error loss)
Given x = 0.5, y = 1, β = 0.9, η = 0.1, u₈ = 0.8, w₉ = 2.5, b₉ = 0:
\begin{aligned}
w_{la}&=w_9-\beta u_8=2.5-0.72=1.78,\qquad b_{la}=0-0.72=-0.72\\
\hat y&=\sigma(1.78\cdot 0.5-0.72)=\sigma(0.17)\approx 0.5424\\
\nabla&=(\hat y-y)\,\hat y(1-\hat y)\,x=(-0.4576)(0.2482)(0.5)\approx -0.0568\\
u_9&=\beta u_8+\nabla=0.72-0.0568=0.6632\\
w_{10}&=w_9-\eta u_9=2.5-0.06632\approx 2.4337
\end{aligned}
An answer key may show −0.056 for the gradient because of rounding. Note that 1.78 and −0.72 are the look-ahead point, not w₁₀ and b₁₀.
C3. Quick self-check list
• Write the PLA update rule and say why it converges.
• Build XOR with the 4-hidden-unit network and state the output inequalities.
• Derive the gradient for w of a sigmoid neuron with squared error.
• Derive the gradient of the loss at a_L: ŷᵢ − 1 if i is the true class, ŷᵢ otherwise.
• Write the full backprop loop with shapes.
• Contrast momentum, NAG, SGD and mini-batch, in both update rule and behaviour.
