These equations are the gradients of the squared-error loss for a single neuron with a sigmoid activation function.

Let's derive them from first principles, connecting the neuron, its activation, the loss function and the chain rule.

## 1. The neuron model

Consider a neuron with:

- Input: $x$
- Weight: $w$
- Bias: $b$

First, calculate the weighted input:

$$ z=wx+b $$

Pass this through the sigmoid activation function:

$$ f(x)=\sigma(z)=\frac{1}{1+e^{-z}} $$

The neuron produces a prediction:

$$ \hat y=f(x) $$

Our objective is to adjust $w$ and $b$ so that the prediction approaches the actual output $y$.

## 2. Define the loss function

Let us use squared-error loss:

$$ L=\frac{1}{2}(f(x)-y)^2 $$

The factor $1/2$ is included to simplify differentiation.

We want to find:

$$ \frac{\partial L}{\partial w} \qquad \text{and} \qquad \frac{\partial L}{\partial b} $$

These tell us how sensitive the loss is to changes in weight and bias.

## 3. Apply the chain rule

Notice that the loss depends on $w$ indirectly through the sigmoid output.

The dependency is:

Weight and bias

$w,\ b$

Weighted input

$z=wx+b$

Activation

$f(x)=\sigma(z)$

Loss

$L=\frac12(f(x)-y)^2$

The chain rule gives:

$$ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial z} \cdot \frac{\partial z}{\partial w} $$

We calculate these three derivatives separately.

## 4. Calculate the three derivatives

A. Derivative of loss with respect to prediction

$$ L=\frac12(f(x)-y)^2 $$

Differentiating:

$$ \boxed{\frac{\partial L}{\partial f}=f(x)-y} $$

This is the prediction error.

B. Derivative of sigmoid activation

Recall:

$$ f(x)=\frac{1}{1+e^{-z}} $$

Differentiating with respect to $z$:

$$ \frac{\partial f}{\partial z} = \frac{e^{-z}}{(1+e^{-z})^2} $$

Rearranging in terms of sigmoid output:

$$ \boxed{\frac{\partial f}{\partial z}=f(x)(1-f(x))} $$

This is the derivative of the sigmoid function.

C. Derivative of weighted input with respect to weight

$$ z=wx+b $$

Therefore:

$$ \boxed{\frac{\partial z}{\partial w}=x} $$

Similarly:

$$ \boxed{\frac{\partial z}{\partial b}=1} $$

## 5. Combine using the chain rule

For weight:

$$ \frac{\partial L}{\partial w} = \underbrace{(f(x)-y)}\_{\text{prediction error}} \cdot \underbrace{f(x)(1-f(x))}\_{\text{sigmoid derivative}} \cdot \underbrace{x}\_{\text{input}} $$

Hence:

$$ \boxed{ \nabla_w L= (f(x)-y)f(x)(1-f(x))x } $$

For bias:

$$ \frac{\partial L}{\partial b} = (f(x)-y)f(x)(1-f(x))\cdot 1 $$

Hence:

$$ \boxed{ \nabla_b L= (f(x)-y)f(x)(1-f(x)) } $$

These are precisely the two equations you provided. Strictly speaking, these are partial derivatives of the loss $L$, rather than gradients of the weight and bias themselves.

## 6. What does deep learning do with these gradients?

The gradients tell us the direction in which the loss increases.

Gradient descent moves in the opposite direction:

$$ w\_{\text{new}}=w\_{\text{old}}-\eta\frac{\partial L}{\partial w} $$

$$ b\_{\text{new}}=b\_{\text{old}}-\eta\frac{\partial L}{\partial b} $$

where $\eta$ is the learning rate.

This process is repeated over training observations to reduce the loss.

## 7. The deeper connection: Backpropagation

Notice that our derivation has three components:

| Component          | Role                                            |
| ------------------ | ----------------------------------------------- |
| $f(x)-y$       | Error at the output                             |
| $f(x)(1-f(x))$ | How much the activation changes                 |
| $x$            | How much the weighted input changes with weight |

This is the fundamental idea behind backpropagation.

For a network with many layers, the same chain-rule principle is applied repeatedly, propagating the loss gradient backwards through the network.

One important distinction: these equations assume squared-error loss. If we use sigmoid activation with binary cross-entropy loss, the sigmoid derivative cancels with part of the loss derivative, giving the simpler output gradient:

$$ \frac{\partial L}{\partial z}=f(x)-y $$

That is one reason sigmoid plus binary cross-entropy is commonly used for binary classification.

The central idea: Backpropagation is not a new differentiation technique. It is the systematic application of the chain rule to a network of interconnected functions.
