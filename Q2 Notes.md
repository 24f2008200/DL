# Deep Learning: Exam Notes & Formula Sheet

Lectures 5, 6, 7, 11, 13 (CS6910 / CS7015 slides; history omitted) Prepared for B Nadar  
Lecture 5 = optimizers  $\cdot$  Lecture 6 = bias/variance & regularization  $\cdot$  Lecture 7 = pre-training & activations  $\cdot$  Lecture 11 = CNNs  $\cdot$  Lecture 13 = visualizing CNNs

> Math is written in LaTeX (`$...$` inline, `$$...$$` display) and renders natively on GitHub.

---

# LECTURE 5: LEARNING PARAMETERS (OPTIMIZERS)

## 5.1 Gradient descent rule

Taylor: $L(\theta+\eta u)=L(\theta)+\eta\,u^{T}\nabla L(\theta)+\frac{\eta^2}{2!}u^{T}\nabla^2L(\theta)u+\cdots$. We need $u^{T}\nabla L<0$; with $u^{T}\nabla L=k\cos\beta$ the most negative value is at $\beta=180^\circ$, so 

$$
\theta_{t+1}=\theta_t-\eta\,\nabla L(\theta_t)
$$

Gradient is large on steep slopes and tiny on gentle slopes, so plain GD is slow on gentle regions. Contours: closely spaced = steep, widely spaced = gentle.

## 5.2 Momentum and NAG

$$
\text{Momentum: }\ u_t=\beta u_{t-1}+\nabla w_t,\quad w_{t+1}=w_t-\eta\,u_t,\quad u_{-1}=0,\ 0\le\beta<1,\qquad u_t=\sum_{\tau=0}^{t}\beta^{\,t-\tau}\nabla w_\tau
$$



$$
\text{NAG: }\ u_t=\beta u_{t-1}+\nabla\bigl(w_t-\beta u_{t-1}\bigr),\quad w_{t+1}=w_t-\eta\,u_t
$$

- Momentum: faster on gentle slopes, but overshoots and makes U-turns across a valley. NAG “looks before it leaps”: smaller oscillations, shorter U-turns, corrects course sooner.

- The slide code for NAG uses $\eta$ inside the update ($v=\beta v+\eta\nabla$, $w=w-v$); the lecture formula puts $\eta$ outside as above. Follow the form given in the question.

## 5.3 Batch, stochastic and mini-batch

| Algorithm     | Steps per epoch |     Gradient      | Behaviour                                                    |
|:--------------|:---------------:|:-----------------:|:-------------------------------------------------------------|
| Batch GD      |       $1$       | exact (full data) | loss falls each step; slow for large $N$                     |
| SGD           |       $N$       | one point (noisy) | fast updates; many oscillations; no per-step guarantee       |
| Mini-batch GD |      $N/B$      |    $B$ points     | fewer oscillations (e.g. $B=25$); larger $B$ = more accurate |

1 epoch = one pass over the data; 1 step = one parameter update. Stochastic and mini-batch versions of momentum and NAG exist; NAG still takes shorter U-turns, and both beat plain SGD.

## 5.4 Learning-rate tips and schedules

- Initial $\eta$: try $0.0001,0.001,0.01,0.1,1.0$ for a few epochs, then search finely around the best (heuristic, no clear winner).

- Step decay: halve $\eta$ every 5 epochs, or halve it when validation error rises compared with the previous epoch.

- Exponential decay and $1/t$ decay ($\eta_0,k$ hyperparameters, $t$ = step):

$$
\eta=\eta_0 e^{-kt}\qquad\qquad \eta=\frac{\eta_0}{1+kt}
$$



$$
\text{Momentum schedule (Sutskever 2013): }\ \beta_t=\min\Bigl(1-2^{-1-\log_2(\lfloor t/250\rfloor+1)},\ \beta_{\max}\Bigr),\quad \beta_{\max}\in\{0.999,0.995,0.99,0.9,0\}
$$

- **Line search:** each step, try $\eta\in\{0.1,0.5,1,2,10\}$ and keep the update with the lowest loss. Faster convergence, but many more loss evaluations per step.

## 5.5 Adaptive learning-rate methods

Why: for a sparse feature $x_2$ the gradient $\nabla w_2=(f-y)f(1-f)x_2$ is 0 most of the time, so $w_2$ rarely updates. We want a *per-parameter* learning rate: bigger for infrequent (sparse) features, smaller for frequently updated ones (such as the bias).

**AdaGrad** (decays $\eta$ in proportion to update history): 

$$
v_t=v_{t-1}+(\nabla w_t)^2,\qquad w_{t+1}=w_t-\frac{\eta}{\sqrt{v_t+\epsilon}}\,\nabla w_t
$$

$v_t$ only grows, so the effective rate $\eta/\sqrt{v_t+\epsilon}$ decays monotonically. Frequent parameters (e.g. $b$) decay fast, sparse ones slowly. Drawbacks: decays too aggressively (updates become tiny, slows near minimum), no way to raise $\eta$ later in gentle regions, and large initial gradients leave $\eta$ low for the rest of training. Removing the square root works poorly.

**RMSProp** (decay the denominator with a moving average): 

$$
v_t=\beta v_{t-1}+(1-\beta)(\nabla w_t)^2,\qquad w_{t+1}=w_t-\frac{\eta}{\sqrt{v_t+\epsilon}}\,\nabla w_t
$$

The effective rate can rise, fall or stay constant. A constant rate near the minimum can oscillate. Still sensitive to initial $\eta_0$.

**AdaDelta** (removes the need to set $\eta_0$): 

$$
v_t=\beta v_{t-1}+(1-\beta)(\nabla w_t)^2,\quad
\Delta w_t=-\frac{\sqrt{u_{t-1}+\epsilon}}{\sqrt{v_t+\epsilon}}\,\nabla w_t,\quad
w_{t+1}=w_t+\Delta w_t,\quad
u_t=\beta u_{t-1}+(1-\beta)(\Delta w_t)^2
$$

The numerator $u_{t-1}$ is a running average of past updates; it can grow or shrink with the gradients, so the rate adapts (in a low-curvature region $v_t$ falls and the effective rate grows).

**Adam** (RMSProp/AdaDelta ideas + cumulative gradient history, i.e. momentum): 

$$
\begin{aligned}
m_t&=\beta_1 m_{t-1}+(1-\beta_1)\nabla w_t, & \hat m_t&=\frac{m_t}{1-\beta_1^{\,t}}\\
v_t&=\beta_2 v_{t-1}+(1-\beta_2)(\nabla w_t)^2, & \hat v_t&=\frac{v_t}{1-\beta_2^{\,t}}
\end{aligned}
\qquad
w_{t+1}=w_t-\frac{\eta}{\sqrt{\hat v_t+\epsilon}}\,\hat m_t
$$

Typical: $\beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8}$.

- **Why bias correction:** with $m_0=0$, $\mathbb{E}[m_t] = \mathbb{E}[\nabla w](1-\beta_1^t)$ (geometric series, assuming all gradients share one distribution), so dividing by $1-\beta_1^{\,t}$ gives $\mathbb{E}[\hat m_t]=\mathbb{E}[\nabla w]$.

- Example: $\nabla w_0=0.1$, $\beta_2=0.999$: $v_0=0.001\cdot0.01=10^{-5}$; without correction $1/\sqrt{v_0}\approx316$, with correction $\hat v_0=0.01$ so $1/\sqrt{\hat v_0}=10$. No correction gives huge first steps.

- Practical advice: Adam is more or less the default; for sequence generation $\eta=0.001$ or $0.0001$ works best. SGD with momentum (or NAG) and an annealing schedule also works well. Adam may fail to converge in some cases and often generalizes worse.

**AdaMax** (replace the $L_2$ norm in $v_t$ by the max norm; no bias correction needed for $v_t$): 

$$
v_t=\max\bigl(\beta_2 v_{t-1},\,\lvert\nabla w_t\rvert\bigr),\qquad w_{t+1}=w_t-\frac{\eta}{v_t+\epsilon}\,\hat m_t
$$

(MaxProp is the same idea applied to RMSProp.)

**NAdam** = NAG inside Adam: NAG normally uses momentum twice (once for the gradient look-ahead, once for the step), so rewrite it so the look-ahead is applied once: 

$$
g_{t+1}=\nabla w_t,\qquad m_{t+1}=\beta m_t+g_{t+1},\qquad w_{t+1}=w_t-\bigl(\beta m_{t+1}+g_{t+1}\bigr)\quad(\eta=1\text{ as drawn on the slides})
$$

Other optimizers named in the lecture: AMSGrad, AdamW.

## 5.6 More schedules: cyclical and cosine annealing

- **Cyclical LR (CLR):** saddle points, not bad local minima, are often the real difficulty; an exponentially decayed $\eta$ cannot escape one, so vary $\eta$ cyclically. Triangular form with step size $\mu$:

$$
\eta_t=\eta_{\min}+(\eta_{\max}-\eta_{\min})\cdot\max\Bigl(0,\ 1-\Bigl\lvert\frac{t}{\mu}-2\Bigl\lfloor1+\frac{t}{2\mu}\Bigr\rfloor+1\Bigr\rvert\Bigr)
$$

- **Cosine annealing with warm restarts** ($T$ = restart interval): $\eta_t=\eta_{\max}$ at $t=0$ and $\eta_{\min}$ at $t=T$, then it jumps back up (hence “warm restart”).

$$
\eta_t=\eta_{\min}+\frac{\eta_{\max}-\eta_{\min}}{2}\Bigl(1+\cos\Bigl(\pi\,\frac{t\bmod(T+1)}{T}\Bigr)\Bigr)
$$

- **Warm-up:** start with a low $\eta$ and increase it (e.g. 4000 warm-up steps in transformers), then decay.

# LECTURE 6: BIAS–VARIANCE AND REGULARIZATION

## 6.1 Bias and variance

True model $f(x)$, estimate $\hat f(x)$, data $y=f(x)+\varepsilon$, $\varepsilon\sim\mathcal{N}(0,\sigma^2)$. 

$$
\text{Bias}(\hat f(x))=\mathbb{E}[\hat f(x)]-f(x),\qquad
\text{Var}(\hat f(x))=\mathbb{E}\bigl[(\hat f(x)-\mathbb{E}[\hat f(x)])^2\bigr]
$$



$$
\mathbb{E}\bigl[(y-\hat f(x))^2\bigr]=\text{Bias}^2+\text{Variance}+\sigma^2\quad(\sigma^2=\text{irreducible error})
$$

- Simple model (e.g. degree-1 fit to a sinusoid): high bias, low variance. Complex model (degree-25): low bias, high variance. There is a trade-off; the sweet spot lies in between.

- Variance = how much models trained on different samples differ from each other.

## 6.2 Train error vs. test error and model complexity

$$
\text{train}_{err}=\frac1n\sum_{i=1}^{n}\bigl(y_i-\hat f(x_i)\bigr)^2,\qquad
\text{test}_{err}=\frac1m\sum_{i=n+1}^{n+m}\bigl(y_i-\hat f(x_i)\bigr)^2
$$

Train error becomes overly optimistic as complexity grows; test (validation) error shows the real picture. Derivation: $\mathbb{E}[(\hat f-f)^2]=\mathbb{E}[(\hat y-y)^2]-\sigma^2+2\,\text{Cov}\bigl(\varepsilon,\hat f-f\bigr)$.

- **Test data:** $\varepsilon$ is independent of $\hat f-f$, so Cov $=0$ and true error $=$ empirical test error $-$ small constant. Always estimate error on an independent validation set.

- **Training data:** $\hat f$ was fit to these points, so Cov $\ne0$. By Stein’s lemma it equals $\frac{\sigma^2}{n}\sum_i\frac{\partial\hat f(x_i)}{\partial y_i}$ (sensitivity of the fit to the data; large for complex models).

$$
\text{true error}=\text{empirical train error}+\text{small constant}+\Omega(\text{model complexity}),\qquad
\min_\theta\ \mathcal{L}_{train}(\theta)+\Omega(\theta)
$$

This is the basis of all regularization ($\ell_1$, $\ell_2$, early stopping, input noise are instances). Deep nets are very complex and easily drive training error to 0, hence regularization is needed.

## 6.3 L2 regularization (weight decay)

$$
\tilde L(w)=L(w)+\frac{\alpha}{2}\lVert w\rVert^2,\qquad \nabla\tilde L=\nabla L+\alpha w,\qquad
w_{t+1}=w_t-\eta\,\nabla L(w_t)-\eta\alpha\,w_t
$$

Geometric view: let $w^*$ minimize $L$ (so $\nabla L(w^*)=0$) with Hessian $H=Q\Lambda Q^{T}$. With the quadratic approximation $\nabla L(w)=H(w-w^*)$, setting $\nabla\tilde L(\tilde w)=0$ gives 

$$
(H+\alpha I)\tilde{w}=Hw^*
\;\Rightarrow\;
\tilde{w}=(H+\alpha I)^{-1}Hw^*
=Q(\Lambda+\alpha I)^{-1}\Lambda Q^{T}w^*
$$

Each component of $Q^{T}w^*$ is scaled by $\dfrac{\lambda_i}{\lambda_i+\alpha}$: if $\lambda_i\gg\alpha$ the factor is about 1 (direction kept), if $\lambda_i\ll\alpha$ it is about 0 (direction shrunk). Only important directions (large eigenvalues) keep large weights. Effective number of parameters $=\sum_i\frac{\lambda_i}{\lambda_i+\alpha}<n$.

## 6.4 Other regularization methods

- **Dataset augmentation:** rotate, shift, etc. (works well for images, also speech). More data $\Rightarrow$ better learning; unclear how to do it for some tasks.

- **Parameter sharing:** same filter or weights reused (CNNs). **Parameter tying:** e.g. encoder and decoder weights tied in autoencoders.

- **Noise on inputs:** $\tilde x_i=x_i+\varepsilon_i$, $\varepsilon_i\sim\mathcal{N}(0,\sigma^2)$. For a simple linear net $\tilde y=\sum w_i x_i+\sum w_i\varepsilon_i$ the expected error is (cross terms vanish by independence)

$$
\mathbb{E}[(\tilde y-y)^2]=\mathbb{E}[(\hat y-y)^2]+\sigma^2\sum_{i=1}^{n}w_i^2\qquad(\text{same as an }\ell_2\text{ penalty})
$$

- **Noise on outputs (label smoothing):** do not fully trust labels. Hard target $[0,\dots,1,\dots,0]$ becomes soft: true class gets $1-\epsilon$, each of the other 9 classes (in a 10-class problem) gets $\epsilon/9$. Minimize cross-entropy $-\sum_i p_i\log q_i$ against the soft $p$.

- **Early stopping:** track validation error with a patience $p$; if no improvement in the previous $p$ steps at step $k$, stop and return the model stored at step $k-p$. Very effective, most widely used, combinable with $\ell_2$. Equivalent to weight decay: with $w_0=0$ and SGD,

$$
w_t=Q\bigl[I-(I-\eta\Lambda)^t\bigr]Q^{T}w^*
\qquad \text{(early stopping)}
$$

$$
\tilde{w}=Q\bigl[I-(\Lambda+\alpha I)^{-1}\alpha I\bigr]Q^{T}w^*
\qquad \text{($\ell_2$ regularization)}
$$

equal when $(I-\eta\Lambda)^t=(\Lambda+\alpha I)^{-1}\alpha$. Only $t$ updates are allowed, so parameters of unimportant directions (small gradients) cannot grow large, which shrinks them like weight decay.

## 6.5 Ensembles (bagging)

Bagging: build $T_1,\dots,T_k$ by sampling *with replacement* (about 36% of samples are duplicates), train one model on each, average (regression) or vote/mode (classification). Each model has error $\varepsilon_i$ with variance $V=\mathbb{E}[\varepsilon_i^2]$ and covariance $C=\mathbb{E}[\varepsilon_i\varepsilon_j]$: 

$$
\text{mse}=\mathbb{E}\Bigl[\Bigl(\frac1k\sum_i\varepsilon_i\Bigr)^2\Bigr]=\frac1k V+\frac{k-1}{k}C
$$

Perfectly correlated errors ($V=C$): mse $=V$, no gain. Independent errors ($C=0$): mse $=V/k$. On average the ensemble is at least as good as its members.

## 6.6 Dropout

- Training many large networks and combining them at test time is too expensive. Dropout trains exponentially many “thinned” networks at almost no extra cost: with $n$ nodes there are $2^n$ thinned networks.

- Each node is retained with probability $p$ (typically $0.5$ for hidden nodes, $0.8$ for input/visible nodes). Temporarily remove the node with its incoming and outgoing connections.

- Trick: *share weights* across all networks and *sample a different thinned network* for each training instance (or mini-batch); backprop updates only the active weights. A thinned network is trained rarely, but weight sharing means no network has untrained parameters.

- **Test time:** use the full network and scale each node’s output (equivalently its weights) by $p$, the fraction of times it was on during training. Aggregating $2^n$ networks directly is impossible.

- Effect: masking noise on hidden units; prevents co-adaptation; each unit must be robust, giving redundancy (e.g. if the “nose” unit is dropped, the net must learn another way to detect a face).

## 6.7 Grouping regularization and implicit regularization

- By component: **data** (augmentation, noise injection); **architecture** (dropout, skip connections, weight sharing, pooling); **optimizer** (SGD, early stopping); **penalty cost** ($\ell_1$, $\ell_2$). $\ell_1$ pushes most weights to 0; $\ell_2$ prevents weights from getting large.

- Explicit: $\ell_1,\ell_2$, augmentation, noise, dropout, skip connections, weight sharing, pooling. Implicit: SGD, large or small initial LR, early stopping.

- Random-labelling experiment (Zhang et al.): deep nets reach zero training error even on randomly shuffled but fixed labels. Properly tuned regularizers help, but they are unlikely to be the fundamental reason for generalization, since networks still generalize with all regularizers removed. The objective value alone says little about generalization; over-parameterized nets are implicitly regularized by (S)GD.

# LECTURE 7: TRAINING DEEP NETWORKS, PRE-TRAINING, ACTIVATIONS

## 7.1 Why deep nets were hard, and pre-training

- Training is a game of gradients. The gradient w.r.t. a parameter measures its responsibility for the loss and is proportional to the input it multiplies (the $x$ or $h_{i-1}$ term in $\nabla w_i$).

- **Unsupervised layer-wise pre-training (Hinton & Salakhutdinov 2006):** train layer 1 to reconstruct $x$ from $h^1$ ($\min\frac1m\sum_i\sum_j(\hat x_{ij}-x_{ij})^2$), fix it, train layer 2 to reconstruct $h^1$ ($\min\frac1m\sum\sum(\hat h^1_{ij}-h^1_{ij})^2$), and so on up to the last hidden layer. Then add the output layer and **fine-tune** the whole network with the supervised objective $\arg\min\frac1m\sum_i(\hat y_i-f(x_i))^2$.

- **Why it works:** (i) *optimization:* the supervised error surface is highly non-convex; if the last layer has large capacity the loss reaches 0 even without pre-training, but with small capacity pre-training helps (Larochelle et al.); (ii) *regularization:* the unsupervised objective acts as an extra constraint $\Omega(\theta)=\frac1m\sum\sum(x_{ij}-\hat x_{ij})^2$, restricting weights to regions that capture the data’s characteristics so learning is not greedy w.r.t. the supervised objective.

- Since 2006–09 progress came from better optimizers, better regularization, better activation functions and better weight initialization.

- A deep *linear* network cannot learn non-linear boundaries; a deep non-linear network can (universal approximation).

## 7.2 Activation functions

| Name                    | Formula / derivative                                                                                                                                     | Key points                                                                                                                                                                                                                        |
|:------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Sigmoid                 | $\sigma(x)=\dfrac1{1+e^{-x}}$,  $\sigma'=\sigma(1-\sigma)$                                                                                               | Range $(0,1)$. Saturates at 0 or 1 where the gradient is 0 (vanishing gradient; also from large initial weights). Not zero-centred: all gradients at a layer have the same sign.                                                  |
| tanh                    | $\tanh x\in[-1,1]$,  $\tanh'=1-\tanh^2$                                                                                                                  | Zero-centred, but still saturates and is expensive (exp).                                                                                                                                                                         |
| ReLU                    | $f(x)=\max(0,x)$;  $f'=1$ for $x>0$, $0$ for $x<0$                                                                                                       | Does not saturate for $x>0$, cheap, converges much faster. A large negative bias gives a *dead neuron* (zero gradient forever); can happen with a high LR. Fix: initialize bias to a small positive value (0.01) or use variants. |
| Leaky / Parametric ReLU | $\max(0.1x,x)$;  PReLU: $\max(\alpha x,x)$, $\alpha$ learned                                                                                             | No saturation, never dies, cheap, near zero-centred.                                                                                                                                                                              |
| ELU                     | $x$ if $x>0$;  $\alpha(e^{x}-1)$ if $x\le0$                                                                                                              | All ReLU benefits, small gradient for $x\le0$, near zero-centred outputs; expensive (exp).                                                                                                                                        |
| Maxout                  | $\max(w_1^{T}x+b_1,\dots,w_n^{T}x+b_n)$                                                                                                                  | Generalizes ReLU and Leaky ReLU; no saturation, no death; $k$ times the parameters. Two Maxout units can act as a universal approximator.                                                                                         |
| GELU                    | $f(x)=m\,x,\ m\sim\text{Bernoulli}(\Phi(x))$; $\mathbb{E}[f]=x\Phi(x)\approx0.5x\bigl(1+\tanh[\sqrt{2/\pi}(x+0.044715x^3)]\bigr)\approx x\sigma(1.702x)$ | Stochastic ReLU/dropout-like: multiplies the input by 1 or 0 with probability depending on the input. Common in transformers (BERT, GPT).                                                                                         |
| SELU                    | $\lambda x$ if $x>0$;  $\lambda\alpha(e^{x}-1)$ if $x\le0$,  $\lambda\approx1.0507$, $\alpha\approx1.6733$                                               | Self-normalizing: keeps activations near zero mean and unit variance.                                                                                                                                                             |
| Swish / SiLU            | $x\,\sigma(\beta x)$                                                                                                                                     | Found by automatic search. $\beta=1.702$ gives GELU; $\beta=1$ gives SiLU.                                                                                                                                                        |

Remember: sigmoids are bad; ReLU is more or less the standard for CNNs (also try Leaky ReLU, Maxout, ELU); tanh and sigmoid are still used in LSTMs/RNNs; GELU is most common in transformers.

# LECTURE 11: CONVOLUTIONAL NEURAL NETWORKS

## 11.1 Convolution operation

1D (weighted average over a window, filter $w$): $s_t=\sum_{a=0}^{6}x_{t-a}\,w_{-a}$. 2D with an $m\times n$ kernel $K$ (as used in practice, looking at succeeding neighbours, or centred on the pixel): 

$$
S_{ij}=(I*K)_{ij}=\sum_{a=0}^{m-1}\sum_{b=0}^{n-1}I_{i+a,\,j+b}\,K_{a,b}
$$

Sliding the kernel over the image gives one value per position: a **feature map**. Many filters give many feature maps. The filter’s depth equals the input depth. Example kernels: sharpen $\begin{pmatrix}0&-1&0\\-1&5&-1\\0&-1&0\end{pmatrix}$; the blur and edge kernels are other examples.

## 11.2 Output size

Input $W_1\times H_1\times D_1$; $K$ filters of size $F\times F$ (depth $D_1$); stride $S$; zero-padding $P$: 

$$
W_2=\frac{W_1-F+2P}{S}+1,\qquad H_2=\frac{H_1-F+2P}{S}+1,\qquad D_2=K
$$

Without padding and with $S=1$: $W_2=W_1-F+1$ (output shrinks). A $3\times3$ kernel with $P=1$, $S=1$ keeps the same size. Exercises from the slides: $227\times227\times3$, $F=11$, $S=4$, $P=0$, $K=96$ gives $55\times55\times96$; $32\times32\times1$, $F=5$, $S=1$, $K=6$ gives $28\times28\times6$.

Parameters of a conv layer: $F\cdot F\cdot D_1\cdot K$ (plus $K$ biases). Pooling layers have 0 parameters.

## 11.3 CNN ideas

- A CNN is a feedforward network with **sparse connectivity** (only a few local pixels feed each neuron, so far fewer parameters) and **weight sharing** (the same kernel is applied at all locations; many kernels allowed). Sparse connectivity does not lose interactions: distant pixels interact indirectly in deeper layers.

- It is trained by ordinary backpropagation, treating it as a feedforward network where only a few weights are active (the rest are zero).

- **Pooling:** max pooling (or average pooling), e.g. $2\times2$ with stride 2: $\begin{pmatrix}1&4&2&1\\5&8&3&4\\7&6&4&5\\1&3&1&2\end{pmatrix}\to\begin{pmatrix}8&4\\7&5\end{pmatrix}$. With stride 1 the output is larger ($3\times3$).

## 11.4 LeNet-5 parameter count (from the slides)

Input $32\times32$; Conv1 ($K=6$, $F=5$, $S=1$) $\to28\times28\times6$; pool ($F=2$, $S=2$) $\to14\times14\times6$; Conv2 ($K=16$, $F=5$) $\to10\times10\times16$; pool $\to5\times5\times16$; FC 120; FC 84; output 10.

| Layer   |           Params            | Layer                      |         Params          |
|:--------|:---------------------------:|:---------------------------|:-----------------------:|
| Conv1   |  $5\cdot5\cdot1\cdot6=150$  | FC1 (400$\to$120)  | $400\cdot120+120=48120$ |
| Conv2   | $5\cdot5\cdot6\cdot16=2400$ | FC2 (120$\to$84)   |  $120\cdot84+84=10164$  |
| Pooling |              0              | Output (84$\to$10) |   $84\cdot10+10=850$    |

(The slide counts conv parameters without biases and FC parameters with biases.)

## 11.5 ImageNet architectures

| Year (ILSVRC) | Model, top-5 error          | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|:--------------|:----------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2012          | AlexNet, 16.4% (8 layers)   | Input $227\times227\times3$; Conv1: 96 filters $11\times11$, stride 4 $\to55\times55\times96$; max-pool; $\ldots$; flattened conv output feeds dense 4096–4096–1000 layers; 27.55M parameters; ReLU.                                                                                                                                                                                                                                                                                                                                                            |
| 2013          | ZFNet, 11.7% (8 layers)     | Same layout as AlexNet but a smaller first filter ($7\times7$) and more filters in later layers (1.45M more parameters).                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 2014          | VGG, 7.3% (19 layers)       | $3\times3$ kernels throughout. About 16M parameters in non-FC layers, about 122M in FC layers; most are in the first FC layer ($512\cdot7\cdot7\cdot4096\approx102$M).                                                                                                                                                                                                                                                                                                                                                                                          |
| 2014          | GoogLeNet, 6.7% (22 layers) | **Inception module**: apply $1\times1$, $3\times3$, $5\times5$ convolutions and max-pooling in parallel and concatenate the feature maps. To cut computation, first use $1\times1$ convolutions (aggregate along depth, $D\to D_1<D$) so the cost per output element drops from $O(F\cdot F\cdot D)$ to $O(F\cdot F\cdot D_1)$; max-pooling is followed by a $1\times1$ conv. Removes the big FC layer: $7\times7$ average pooling on each of 1024 maps gives a 1024-vector, avoiding $7\cdot7\cdot1024\cdot1000=49$M parameters. Dropout (40%) before softmax. |
| 2015          | ResNet, 3.57% (152 layers)  | Very deep plain nets train worse (higher error) even though a deeper net could just learn identity in the extra layers. Residual block learns $F(x)$ and outputs $H(x)=F(x)+x$ (identity skip connection retains a copy of the input). Tricks: batch norm after every conv, Xavier/2 initialization, SGD + momentum 0.9, LR 0.1 divided by 10 on plateau, mini-batch 256, weight decay $10^{-5}$, no dropout. Won all five main ImageNet/COCO tracks.                                                                                                           |

Earlier ILSVRC winners were shallow: 28.2% (2010), 25.8% (2011).

# LECTURE 13: VISUALIZING AND UNDERSTANDING CNNs

### 13.1 Patches that maximally activate a neuron

For neurons in a layer (e.g. pool5), find the image patches that make them fire. Individual neurons respond to faces, dog faces, flowers, digits, houses, shiny surfaces.

### 13.2 Visualizing filters

The unit-norm input that maximally excites a neuron is the normalized weight vector: 

$$
\max_{x}\ w^{T}x\ \ \text{s.t.}\ \ \lVert x\rVert^2=x^{T}x=1\quad\Longrightarrow\quad x=\frac{w}{\sqrt{w^{T}w}}
$$

Filters act as pattern detectors and can be plotted as images, but this is interpretable only for the *first* conv layer (later filters act on feature maps, not on pixels).

### 13.3 Occlusion experiments

Mask a region of the input and watch how the predicted class probability changes; a large drop means that region is important for the class.

### 13.4 Influence of input pixels via backprop

The gradient $\partial h_j/\partial x_i$ tells how much input pixel $x_i$ influences neuron $h_j$ (0 = no influence, large = high). Backpropagate to the input and display the gradient matrix as an image. Plain backprop gives blurry influence maps.

### 13.5 Guided backpropagation

Forward pass, keep one neuron in a feature map and set all other neurons of that layer to zero, then backpropagate to the input. ReLU in the forward pass blocks negative values and in the backward pass blocks gradients of dead ReLUs; guided backprop *also* blocks negative gradients, keeping only positive influences, which gives sharper pictures.

### 13.6 Optimization over images

Keep the trained weights fixed and treat the image pixels as parameters: 

$$
\arg\max_{I}\ \bigl(S_c(I)-\lambda\,\Omega(I)\bigr)
$$

$S_c(I)$ = score of class $c$ before softmax; $\Omega$ = regularizer so $I$ looks like an image. Start from a zero or random image and update pixels by gradient ascent. Starting from different initializations (and focusing on different layers or neurons) gives different insights.

### 13.7 Creating images from embeddings

Treat the fc7 layer as an embedding $\phi_0$ of an image of interest and find $x$ with a similar embedding that also looks natural; start from a random (or zero) image and repeat forward pass, compute, update: 

$$
L(x)=\lVert\phi(x)-\phi_0\rVert^2+\lambda\,\Omega(x),\qquad x\leftarrow x-\eta\,\frac{\partial L}{\partial x}
$$

### 13.8 Deep Dream

Start from a real image, pick a layer, and modify the image (gradient ascent on pixels) so the chosen neurons fire even more. The network has learned patterns (dogs, cats, birds) frequent in ImageNet and starts to “see” them even where they hardly exist.

### 13.9 Deep Art (style transfer)

Generated image $\vec x$ should match a content image $\vec p$ in content and a style image $\vec a$ in style. *Content:* activations of the layers should match. *Style:* captured by Gram matrices of activations (if $V\in\mathbb{R}^{64\times(256\cdot256)}$ is a layer’s activation, $V^{T}V$ captures style; deeper layers capture more style), with style loss $L_{style}=\sum_\ell w_\ell E_\ell$ built from $\bigl(G^\ell_{ij}-A^\ell_{ij}\bigr)^2$: 

$$
L_{total}(\vec p,\vec a,\vec x)=\alpha\,L_{content}(\vec p,\vec x)+\beta\,L_{style}(\vec a,\vec x)
$$

### 13.10 Fooling ConvNets

Add a tiny, human-imperceptible perturbation (optimize the image to maximize a wrong class, e.g. “ostrich”) and the net changes its prediction (Szegedy et al. 2013). Starting from random images one can get 99.6% confident predictions for unrecognizable images (Nguyen et al. 2014). Reason: images live in a huge space ($\mathbb{R}^{227\times227}$), only a few points are real images seen in training, and the decision boundaries also decide the vast unseen regions.

# QUICK EXAM TRAPS

- NAG gradient is evaluated at the look-ahead $w_t-\beta u_{t-1}$; momentum at $w_t$. $\eta$ multiplies $u_t$ in the update (course formula).

- AdaGrad: $v_t$ is a *sum* of squared gradients (only grows). RMSProp/AdaDelta/Adam: *moving average* with $(1-\beta)$.

- Adam needs bias correction for *both* $m_t$ ($1-\beta_1^t$) and $v_t$ ($1-\beta_2^t$); AdaMax needs it only for $m_t$.

- AdaDelta has no $\eta_0$; its numerator is $\sqrt{u_{t-1}+\epsilon}$, using the *previous* $u$.

- Bias$^2$ + Variance + $\sigma^2$; simple $=$ high bias, complex $=$ high variance. Test error is unbiased, train error is optimistic.

- $\ell_2$ update: extra term $-\eta\alpha w_t$. Shrinkage factor per eigen-direction is $\lambda_i/(\lambda_i+\alpha)$.

- Input noise with variance $\sigma^2$ acts like $\ell_2$ penalty $\sigma^2\sum w_i^2$. Early stopping acts like weight decay.

- Dropout: train with thinned net, test with full net scaled by $p$; $p=0.5$ hidden, $0.8$ input.

- Bagging: mse $=\frac1kV+\frac{k-1}{k}C$.

- Conv output: $\frac{W_1-F+2P}{S}+1$ and depth $=K$; conv params $F\cdot F\cdot D_1\cdot K$; pooling has none.

- ReLU dead neuron: large negative bias; use bias $0.01$, Leaky ReLU or ELU. Sigmoid: saturation $\Rightarrow$ vanishing gradient; not zero-centred.

- ResNet: $H(x)=F(x)+x$. GoogLeNet: $1\times1$ conv for dimensionality reduction, global average pooling instead of FC.
