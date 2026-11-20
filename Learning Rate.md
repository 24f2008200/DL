# Adaptive Optimizers: Behaviour of Learning Rate Techniques

Notation: $g_t = \nabla_\theta L(\theta_t)$, learning rate $\eta$, small constant $\epsilon$.

## AdaGrad

$$v_t = \sum_{i=1}^{t} g_i^2, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}}\, g_t$$

- $v_t$ only grows, so the effective rate $\eta/\sqrt{v_t+\epsilon}$ only shrinks.
- **Dense features** (frequent, large gradients): $v_t$ grows fast, so the rate decays quickly.
- **Sparse features** (rarely updated): $v_t$ stays small, so the rate decays slowly.
- Downside: the rate can shrink to nearly zero and training stalls.

## RMSProp

$$v_t = \beta v_{t-1} + (1-\beta)\, g_t^2, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}}\, g_t$$

- Exponential moving average, so old gradients are forgotten.
- The effective rate is **not** guaranteed to be non-increasing: if recent gradients get smaller, $v_t$ drops and the rate rises again.

## AdaDelta

$$E[g^2]_t = \rho\, E[g^2]_{t-1} + (1-\rho)\, g_t^2$$

$$\Delta\theta_t = -\frac{\text{RMS}[\Delta\theta]_{t-1}}{\text{RMS}[g]_t}\, g_t, \qquad \text{RMS}[x]_t = \sqrt{E[x^2]_t + \epsilon}$$

- The units of the update match the units of the parameters, so **no global learning rate $\eta_0$ is needed**.
- Hence AdaDelta is not sensitive to an initial learning rate.

## Adam

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2$$

$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}, \qquad \theta_{t+1} = \theta_t - \frac{\eta\, \hat m_t}{\sqrt{\hat v_t} + \epsilon}$$

- Momentum (first moment) plus RMSProp-style scaling (second moment), with bias correction for early steps.

## AdaMax

$$u_t = \max(\beta_2 u_{t-1},\, |g_t|), \qquad \theta_{t+1} = \theta_t - \frac{\eta\, \hat m_t}{u_t}$$

- Replaces the $L^2$-based $v_t$ of Adam with an $L^\infty$ (max-norm) quantity.
- More stable when gradients have occasional large spikes.

## Statement Check

| # | Statement | Verdict |
|---|-----------|---------|
| 1 | AdaGrad: rate decays slowly for dense features | False |
| 2 | AdaGrad: rate decays slowly for sparse features | **True** |
| 3 | RMSProp: rate guaranteed non-increasing | False |
| 4 | AdaDelta is sensitive to initial learning rate $\eta_0$ | False |
| 5 | Adam's $v_t$ is $L^2$-based, AdaMax uses $L^\infty$ | **True** |

**Answer: statements 2 and 5.**
