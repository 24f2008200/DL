Here is the same content in LaTeX.

**Notation correction (your pasted version):**

$$
\frac{\partial L}{\partial a_{L,i}} = \hat{y}_i - \mathbb{1}_{\ell = i}, \qquad \nabla_{a_L} L = \hat{y} - e_\ell
$$

**General recipe**

$$
\nabla_{a_L} L = J_O^{\top}\, \nabla_{\hat{y}} L, \qquad \hat{y} = O(a_L)
$$

where $J_O$ is the Jacobian of the output activation.

---

**1. Softmax + cross-entropy (multi-class)**

$$
L = -\sum_k y_k \log \hat{y}_k, \qquad \frac{\partial L}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i}
$$

$$
\frac{\partial \hat{y}_i}{\partial a_j} = \hat{y}_i(\delta_{ij} - \hat{y}_j)
$$

$$
\boxed{\nabla_{a_L} L = \hat{y} - y} \qquad (\text{one-hot } y:\ \hat{y} - e_\ell)
$$

---

**2. Sigmoid + binary cross-entropy**

$$
L = -\left[y \log \hat{y} + (1-y)\log(1-\hat{y})\right], \qquad \sigma'(a) = \hat{y}(1-\hat{y})
$$

$$
\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}
$$

$$
\boxed{\frac{\partial L}{\partial a} = \hat{y} - y}
$$

For $K$ independent binary outputs (multi-label), this applies per output: $\nabla_{a_L} L = \hat{y} - y$.

---

**3. Identity output + squared error**

$$
L = \tfrac{1}{2}\sum_i (\hat{y}_i - y_i)^2, \qquad \hat{y} = a
$$

$$
\boxed{\nabla_{a_L} L = \hat{y} - y}
$$

Without the $\tfrac{1}{2}$, the gradient is $2(\hat{y} - y)$.

---

These three are the **canonical pairs** (negative log-likelihood of categorical, Bernoulli, and Gaussian models with their natural links), which is why the gradient is always prediction minus target.

**Mismatched combinations (no cancellation)**

**4. Sigmoid + squared error**

$$
L = \tfrac{1}{2}(\hat{y} - y)^2 \;\Rightarrow\; \frac{\partial L}{\partial a} = (\hat{y} - y)\,\hat{y}(1-\hat{y})
$$

The factor $\hat{y}(1-\hat{y}) \to 0$ when the neuron saturates, so learning stalls even for badly wrong predictions.

**5. Softmax + squared error**

$$
L = \tfrac{1}{2}\sum_k (\hat{y}_k - y_k)^2 \;\Rightarrow\; \frac{\partial L}{\partial a_i} = \hat{y}_i\left[(\hat{y}_i - y_i) - \sum_k \hat{y}_k(\hat{y}_k - y_k)\right]
$$

**6. Tanh + squared error**

$$
\frac{\partial L}{\partial a} = (\hat{y} - y)\,(1 - \hat{y}^2)
$$

---

**Summary table**

| Output activation | Loss | $\nabla_{a_L} L$ |
|---|---|---|
| Softmax | Cross-entropy | $\hat{y} - y$ |
| Sigmoid | Binary cross-entropy | $\hat{y} - y$ |
| Identity | Squared error ($\tfrac12$) | $\hat{y} - y$ |
| Sigmoid | Squared error | $(\hat{y} - y) \odot \hat{y} \odot (1 - \hat{y})$ |
| Tanh | Squared error | $(\hat{y} - y) \odot (1 - \hat{y}^2)$ |
| Softmax | Squared error | $\hat{y} \odot \left[(\hat{y} - y) - \langle \hat{y},\, \hat{y} - y\rangle \mathbf{1}\right]$ |

The numerical check from my previous message applies unchanged, since only the formatting differs.
