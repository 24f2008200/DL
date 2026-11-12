# Derivatives of Common Activation Functions in Neural Networks

An activation function introduces non-linearity into a neural network. During backpropagation, we need its derivative to calculate how much the neuron's output changes with respect to its input.

Let the neuron's pre-activation input be:

\$$ z = \sum\_{i=1}^{n} w_i x_i+b \$$

and its output be:

\$$ a=f(z) \$$

The derivative required during backpropagation is:

\$$ \boxed{\frac{da}{dz}=f'(z)} \$$

Let us derive the commonly used activation functions.

## 1. Sigmoid

Common in binary classification

Definition

\$$ \sigma(z)=\frac{1}{1+e^{-z}} \$$

Derivation

Using the chain rule:

\$$ \begin{aligned} \sigma'(z) &=\frac{d}{dz}(1+e^{-z})^{-1}\\\ &=-(1+e^{-z})^{-2}(-e^{-z})\\\ &=\frac{e^{-z}}{(1+e^{-z})^2} \end{aligned} \$$

Now multiply and divide by \$1+e^{-z}\$:

\$$ \sigma'(z) =\frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} \$$

Since

\$$ \sigma(z)=\frac{1}{1+e^{-z}} \$$

and

\$$ 1-\sigma(z)=\frac{e^{-z}}{1+e^{-z}} \$$

we get:

Final derivative

\$$ \boxed{\sigma'(z)=\sigma(z)(1-\sigma(z))} \$$

Important: We can calculate the derivative directly from the already calculated output. No need to calculate the exponential again.

## 2. Tanh (Hyperbolic tangent)

Definition

\$$ f(z)=\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}} \$$

An alternative expression is:

\$$ \tanh(z)=\frac{e^{2z}-1}{e^{2z}+1} \$$

Derivation

Using the quotient rule:

\$$ \begin{aligned} f'(z) &=\frac{2e^{2z}(e^{2z}+1) -2e^{2z}(e^{2z}-1)} {(e^{2z}+1)^2}\\\ &=\frac{4e^{2z}}{(e^{2z}+1)^2} \end{aligned} \$$

Now observe that:

\$$ 1-\tanh^2(z) = 1-\left(\frac{e^{2z}-1}{e^{2z}+1}\right)^2 \$$

\$$ =\frac{(e^{2z}+1)^2-(e^{2z}-1)^2} {(e^{2z}+1)^2} \$$

\$$ =\frac{4e^{2z}}{(e^{2z}+1)^2} \$$

Therefore:

\$$ \boxed{f'(z)=1-\tanh^2(z)} \$$

Or, using the output \$a\$:

\$$ \boxed{f'(z)=1-a^2} \$$

## 3. ReLU (Rectified Linear Unit)

Widely used in hidden layers

Definition

\$$ f(z)=\max(0,z) \$$

Equivalently:

\$$ f(z)= \begin{cases} 0 & z\leq 0\\\ z & z>0 \end{cases} \$$

Derivation

Differentiate each branch:

\$$ f'(z)= \begin{cases} 0 & z<0\\\ 1 & z>0 \end{cases} \$$

At \$z=0\$, the derivative is mathematically undefined because the left and right derivatives differ.

In practice, implementations conventionally assign derivative zero at zero.

\$$ \boxed{ f'(z)= \begin{cases} 0 & z\leq 0\\\ 1 & z>0 \end{cases} } \$$

## 4. Leaky ReLU

This addresses the problem of ReLU neurons becoming inactive for negative inputs.

Definition

\$$ f(z)= \begin{cases} \alpha z & z<0\\\ z & z\geq 0 \end{cases} \$$

Here, \$\alpha\$ is a small positive constant, often 0.01.

Derivation

Differentiating each branch:

\$$ \boxed{ f'(z)= \begin{cases} \alpha & z<0\\\ 1 & z>0 \end{cases} } \$$

At zero, the derivative is undefined when \$\alpha\neq 1\$. Implementations choose a convention.

## 5. ELU (Exponential Linear Unit)

Definition

\$$ f(z)= \begin{cases} z & z>0\\\ \alpha(e^z-1) & z\leq 0 \end{cases} \$$

Derivation

For the positive branch:

\$$ f'(z)=1 \$$

For the negative branch:

\$$ f'(z)=\alpha e^z \$$

Therefore:

\$$ \boxed{ f'(z)= \begin{cases} 1 & z>0\\\ \alpha e^z & z<0 \end{cases} } \$$

Notice that for the negative branch:

\$$ f'(z)=f(z)+\alpha \$$

This is useful because the output can be reused to calculate the derivative.

## 6. Softmax

Common in multiclass classification

Unlike the previous activation functions, Softmax operates on a vector of inputs.

Definition

For \$K\$ output neurons:

\$$ a_i=\frac{e^{z_i}}{\sum\_{j=1}^{K}e^{z_j}} \$$

We need the partial derivative of output \$a_i\$ with respect to input \$z_j\$.

Let:

\$$ S=\sum\_{k=1}^{K}e^{z_k} \$$

Then:

\$$ a_i=\frac{e^{z_i}}{S} \$$

Using the quotient rule:

\$$ \frac{\partial a_i}{\partial z_j} = \frac{ \delta\_{ij}e^{z_i}S-e^{z_i}e^{z_j} }{S^2} \$$

where \$\delta\_{ij}\$ is the Kronecker delta:

\$$ \delta\_{ij}= \begin{cases} 1 & i=j\\\ 0 & i\neq j \end{cases} \$$

Substituting \$a_i\$ and \$a_j\$:

\$$ \boxed{ \frac{\partial a_i}{\partial z_j} =a_i(\delta\_{ij}-a_j) } \$$

This gives two cases:

When \$i=j\$:

\$$ \boxed{\frac{\partial a_i}{\partial z_i}=a_i(1-a_i)} \$$

When \$i\neq j\$:

\$$ \boxed{\frac{\partial a_i}{\partial z_j}=-a_i a_j} \$$

The complete derivative is a Jacobian matrix:

\$$ \boxed{ J={diag}(a)-aa^T } \$$

The important distinction is that Softmax outputs are interdependent. Changing one input affects all output probabilities.

## 7. GELU (Gaussian Error Linear Unit)

Used extensively in Transformer architectures.

Definition

\$$ f(z)=z\Phi(z) \$$

where \$\Phi(z)\$ is the standard normal cumulative distribution function.

Using the product rule:

\$$ f'(z)=\Phi(z)+z\Phi'(z) \$$

Since the derivative of the standard normal CDF is its probability density function:

\$$ \Phi'(z)=\frac{1}{\sqrt{2\pi}}e^{-z^2/2} \$$

Therefore:

\$$ \boxed{ f'(z)=\Phi(z)+ \frac{z}{\sqrt{2\pi}}e^{-z^2/2} } \$$

## Summary: Activation functions and their derivatives

| Activation | Function                     | Derivative                   |
| ---------- | ---------------------------- | ---------------------------- |
| Sigmoid    | \$\sigma(z)\$              | \$\sigma(z)(1-\sigma(z))\$ |
| Tanh       | \$\tanh(z)\$               | \$1-\tanh^2(z)\$           |
| ReLU       | \$\max(0,z)\$              | \$0\$ or \$1\$           |
| Leaky ReLU | \$\max(\alpha z,z)\$       | \$\alpha\$ or \$1\$      |
| ELU        | Piecewise exponential        | \$1\$ or \$\alpha e^z\$  |
| Softmax    | \$e^{z_i}/\sum_j e^{z_j}\$ | \$a_i(\delta\_{ij}-a_j)\$  |
| GELU       | \$z\Phi(z)\$               | \$\Phi(z)+z\phi(z)\$       |

## Why these derivatives matter in backpropagation

Recall:

\$$ a=f(z),\qquad z=wx+b \$$

Suppose the loss is \$L\$. By the chain rule:

\$$ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w} \$$

Therefore:

\$$ \boxed{ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot f'(z)\cdot x } \$$

Similarly:

\$$ \boxed{ \frac{\partial L}{\partial b} = \frac{\partial L}{\partial a} \cdot f'(z) } \$$

The activation derivative is the link between the error arriving at a neuron and the gradient passed backward through that neuron.

One important observation: Sigmoid and Tanh derivatives become very small when their inputs are large in magnitude. This can cause the vanishing gradient problem. ReLU avoids this for positive inputs, which is one reason it became popular in deep networks.

# Loss Functions in Neural Networks – Derivations and Derivatives

A loss function measures the difference between the network's prediction and the actual target.

During training, we minimise this loss by adjusting weights and biases using gradient descent.

Let:

\- \$y\$ = actual target
\- \$\hat y\$ = predicted output
\- \$L(y,\hat y)\$ = loss function

The fundamental quantity required for backpropagation is:

\$$ \boxed{\frac{\partial L}{\partial \hat y}} \$$

This tells us how the loss changes when the prediction changes.

Let us derive the commonly used loss functions.

## 1. Mean Squared Error (MSE)

Regression

Definition

For one observation:

\$$ L(y,\hat y)=\frac{1}{2}(\hat y-y)^2 \$$

The factor \$1/2\$ is included to simplify differentiation.

Derivation

\$$ \begin{aligned} \frac{\partial L}{\partial \hat y} &=\frac{1}{2}\cdot 2(\hat y-y)\\\ &=\hat y-y \end{aligned} \$$

Final derivative

\$$ \boxed{\frac{\partial L}{\partial \hat y}=\hat y-y} \$$

For \$m\$ observations, the Mean Squared Error is:

\$$ L=\frac{1}{m}\sum\_{i=1}^{m}(\hat y_i-y_i)^2 \$$

Therefore:

\$$ \boxed{ \frac{\partial L}{\partial \hat y_i} =\frac{2}{m}(\hat y_i-y_i) } \$$

Note that the derivative differs depending on whether we include the \$1/2\$ factor.

## 2. Mean Absolute Error (MAE)

Regression

Definition

\$$ L(y,\hat y)=|\hat y-y| \$$

Using the piecewise definition:

$$
L = \begin{cases}
\hat y-y & \hat y>y\\,
y-\hat y & \hat y<y,
\end{cases}
$$

Derivation

Differentiating:

$$
\boxed{
\frac{\partial L}{\partial \hat y}=
\begin{cases}
1 & \hat y>y\\,
-1 & \hat y<y,
\end{cases}
}
$$

At \$\hat y=y\$, the derivative is undefined. A subgradient of zero is commonly used.

Unlike MSE, the gradient magnitude does not increase with the prediction error.

## 3. Binary Cross-Entropy (BCE)

Binary classification

Suppose the target is either 0 or 1, and the predicted probability of class 1 is \$\hat y\$.

Definition

\$$ L(y,\hat y)= -\left[y\ln\hat y+(1-y)\ln(1-\hat y)\right] \$$

Derivation

Differentiating with respect to \$\hat y\$:

\$$ \begin{aligned} \frac{\partial L}{\partial\hat y} &=-\left[ y\frac{1}{\hat y} +(1-y)\frac{-1}{1-\hat y} \right]\\\ &=-\frac{y}{\hat y} +\frac{1-y}{1-\hat y} \end{aligned} \$$

Taking the common denominator:

\$$ =\frac{-y(1-\hat y)+(1-y)\hat y} {\hat y(1-\hat y)} \$$

Simplifying:

\$$ \boxed{ \frac{\partial L}{\partial\hat y} = \frac{\hat y-y}{\hat y(1-\hat y)} } \$$

This is the derivative with respect to the predicted probability.

Important: BCE combined with sigmoid

Suppose:

\$$ \hat y=\sigma(z) \$$

We know:

\$$ \frac{\partial\hat y}{\partial z} =\hat y(1-\hat y) \$$

Applying the chain rule:

\$$ \begin{aligned} \frac{\partial L}{\partial z} &=\frac{\partial L}{\partial\hat y} \frac{\partial\hat y}{\partial z}\\\ &=\frac{\hat y-y}{\hat y(1-\hat y)} \cdot\hat y(1-\hat y) \end{aligned} \$$

Thus:

Final derivative: BCE + Sigmoid

\$$ \boxed{\frac{\partial L}{\partial z}=\hat y-y} \$$

Notice the cancellation! This is one of the important simplifications in neural network backpropagation.

## 4. Categorical Cross-Entropy (CCE)

Multiclass classification

Used when the target belongs to one of \$K\$ mutually exclusive classes.

Let:

- \$y_i\$ = actual target for class \$i\$ (one-hot encoded).
- \$\hat y_i\$ = predicted probability for class \$i\$.

Definition

\$$ L=-\sum\_{i=1}^{K}y_i\ln\hat y_i \$$

For a one-hot encoded target, only the correct class contributes to the loss.

Derivation

Differentiating with respect to \$\hat y_j\$:

\$$ \boxed{ \frac{\partial L}{\partial\hat y_j} =-\frac{y_j}{\hat y_j} } \$$

CCE combined with Softmax

Recall the Softmax derivative:

\$$ \frac{\partial\hat y_i}{\partial z_j} =\hat y_i(\delta\_{ij}-\hat y_j) \$$

Applying the chain rule:

\$$ \frac{\partial L}{\partial z_j} = \sum\_{i=1}^{K} \frac{\partial L}{\partial\hat y_i} \frac{\partial\hat y_i}{\partial z_j} \$$

Substitute:

\$$ = -\sum\_{i=1}^{K} \frac{y_i}{\hat y_i} \hat y_i(\delta\_{ij}-\hat y_j) \$$

Cancel \$\hat y_i\$:

\$$ =-\sum\_{i=1}^{K}y_i(\delta\_{ij}-\hat y_j) \$$

\$$ =-y_j+\hat y_j\sum\_{i=1}^{K}y_i \$$

Since one-hot targets satisfy:

\$$ \sum\_{i=1}^{K}y_i=1 \$$

we get:

Final derivative: CCE + Softmax

\$$ \boxed{ \frac{\partial L}{\partial z_j} =\hat y_j-y_j } \$$

This is exactly the same elegant result as BCE + Sigmoid.

## 5. Huber Loss

Robust regression

Huber loss combines the properties of MSE and MAE.

It behaves like MSE for small errors and MAE for large errors.

Let the prediction error be:

\$$ e=\hat y-y \$$

Definition

\$$ L(e)= \begin{cases} \frac{1}{2}e^2 & |e|\leq\delta\\\ \delta(|e|-\frac{1}{2}\delta)&|e|>\delta \end{cases} \$$

Here, \$\delta\$ is a threshold controlling the transition.

Derivation

For the first branch:

\$$ \frac{\partial L}{\partial\hat y} =e=\hat y-y \$$

For the second branch:

\$$ \frac{\partial L}{\partial\hat y} =\delta{sign}(e) \$$

Therefore:

\$$ \boxed{ \frac{\partial L}{\partial\hat y} = \begin{cases} \hat y-y & |\hat y-y|\leq\delta\\\ \delta{sign}(\hat y-y) &|\hat y-y|>\delta \end{cases} } \$$

## 6. Hinge Loss

Support Vector Machines

Used primarily for classification with labels \$y\in\\{-1,+1\\}\$.

Let \$s\$ be the model's classification score.

Definition

\$$ L(y,s)=\max(0,1-ys) \$$

Derivation

\$$ \boxed{ \frac{\partial L}{\partial s} = \begin{cases} -y & ys<1\\\ 0 & ys>1 \end{cases} } \$$

At \$ys=1\$, the derivative is undefined; a subgradient can be selected.

Interpretation:

- If \$ys\geq1\$, the prediction satisfies the margin requirement. No loss gradient is needed.
- If \$ys<1\$, the prediction violates the margin requirement, so the model receives a gradient.

## 7. Kullback-Leibler (KL) Divergence

Used to measure the difference between two probability distributions.

Let \$P\$ be the target distribution and \$Q\$ be the predicted distribution.

Definition

\$$ D\_{KL}(P\\|Q) = \sum\_{i=1}^{K}P_i\ln\frac{P_i}{Q_i} \$$

Derivation

Differentiating with respect to \$Q_j\$:

\$$ \begin{aligned} \frac{\partial D\_{KL}}{\partial Q_j} &=P_j\frac{\partial}{\partial Q_j} \left(\ln P_j-\ln Q_j\right)\\\ &=-\frac{P_j}{Q_j} \end{aligned} \$$

Therefore:

\$$ \boxed{ \frac{\partial D\_{KL}}{\partial Q_j} =-\frac{P_j}{Q_j} } \$$

When \$P\$ is fixed, KL divergence differs from cross-entropy only by a constant:

\$$ D\_{KL}(P\\|Q)=H(P,Q)-H(P) \$$

Consequently, their gradients with respect to \$Q\$ are identical.

## 8. Focal Loss

Used in classification problems with significant class imbalance, particularly object detection.

For binary classification, let \$p_t\$ be the probability assigned to the correct class:

\$$ p_t= \begin{cases} \hat y & y=1\\\ 1-\hat y & y=0 \end{cases} \$$

Definition

\$$ L=-\alpha_t(1-p_t)^\gamma\ln p_t \$$

Here:

- \$\alpha_t\$ controls class weighting.
- \$\gamma\$ controls how strongly easy examples are down-weighted.

Differentiating with respect to \$p_t\$:

\$$ \boxed{ \frac{\partial L}{\partial p_t} = \alpha_t\gamma(1-p_t)^{\gamma-1}\ln p_t - \alpha_t\frac{(1-p_t)^\gamma}{p_t} } \$$

The additional factor \$(1-p_t)^\gamma\$ reduces the contribution of examples that are already classified correctly with high confidence.

## Summary: Loss functions and derivatives

| Loss function             | Typical application         | Derivative                              |
| ------------------------- | --------------------------- | --------------------------------------- |
| MSE                       | Regression                  | \$\hat y-y\$ (with \$1/2\$)         |
| MAE                       | Regression                  | \${sign}(\hat y-y)\$     |
| Binary Cross-Entropy      | Binary classification       | \$\frac{\hat y-y}{\hat y(1-\hat y)}\$ |
| Categorical Cross-Entropy | Multiclass classification   | \$-\frac{y_i}{\hat y_i}\$             |
| Huber                     | Robust regression           | Piecewise: error or clipped error       |
| Hinge                     | Margin-based classification | \$-y\$ or \$0\$                     |
| KL Divergence             | Distribution matching       | \$-\frac{P_i}{Q_i}\$                  |
| Focal Loss                | Imbalanced classification   | Depends on \$p_t,\alpha_t,\gamma\$    |

The table gives derivatives with respect to the prediction, except where indicated. For BCE and CCE, the derivatives with respect to logits after sigmoid or softmax are particularly useful.

## The most important connection: Loss + Activation

For backpropagation, we usually need the derivative with respect to the neuron's pre-activation input \$z\$, not merely its output.

These combinations simplify considerably:

| Activation + Loss                | Derivative with respect to \$z\$  |
| -------------------------------- | ----------------------------------- |
| Sigmoid + BCE                    | \$\hat y-y\$                      |
| Softmax + CCE                    | \$\hat{\mathbf y}-\mathbf y\$     |
| Linear + MSE (\$\frac12 e^2\$) | \$\hat y-y\$                      |
| Linear + MAE                     | \${sign}(\hat y-y)\$ |

This leads to a useful general backpropagation expression:

\$$ \boxed{ \frac{\partial L}{\partial w} = \underbrace{\frac{\partial L}{\partial z}}\_{\text{error signal}} \cdot \underbrace{\frac{\partial z}{\partial w}}\_{x} } \$$

For a neuron:

\$$ \boxed{ \frac{\partial L}{\partial w} =\delta x } \$$

where \$\delta=\partial L/\partial z\$.

One important distinction: The loss function measures prediction error; the activation function determines how the neuron transforms its input. Their derivatives work together through the chain rule.

For practical neural-network implementations, BCE and CCE are usually computed directly from logits using numerically stable implementations rather than explicitly calculating probabilities and logarithms separately.
