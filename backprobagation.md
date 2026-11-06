Let us derive backpropagation for a neural network with two hidden layers, starting from the output layer and working backwards towards the input layer.

We will use sigmoid activation and squared-error loss, consistent with our previous derivation.

The important things we will track are:

1. What is calculated during the forward pass?
2. What must be stored for backpropagation?
3. How do we calculate the error (delta) at each layer?
4. How are weights and biases updated?
5. Which calculated values are reused?

# 1. Network architecture

Consider this fully connected neural network:

<img width="849" height="603" alt="image" src="https://github.com/user-attachments/assets/4a3d9d4a-3cb7-4845-b47f-fa93bdb616e3" />

## Neural network with two hidden layers

Illustrative network: three neurons per layer, with fully connected adjacent layers. The number of neurons can vary.

We will call the layers:

| Layer          | Symbol        | Description         |
| -------------- | ------------- | ------------------- |
| Input          | $a^{(0)}$ | Input features      |
| Hidden layer 1 | $a^{(1)}$ | First hidden layer  |
| Hidden layer 2 | $a^{(2)}$ | Second hidden layer |
| Output         | $a^{(3)}$ | Final prediction    |

Let the number of neurons be:

- Input: $n_0$
- Hidden 1: $n_1$
- Hidden 2: $n_2$
- Output: $n_3$

We use $W^{(l)}$ for weights connecting layer $l-1$ to layer $l$, and $b^{(l)}$ for biases of layer $l$.

# 2. Forward propagation

Before we can calculate gradients, we must calculate the outputs of every neuron.

For each layer, we perform two operations:

$$ z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)} $$

$$ a^{(l)}=\sigma(z^{(l)}) $$

where:

$$ \sigma(z)=\frac{1}{1+e^{-z}} $$

Let's expand this layer by layer.

Input layer

$$ a^{(0)}=x $$

The input is already known. No activation calculation is required.

Hidden layer 1

$$ z^{(1)}=W^{(1)}a^{(0)}+b^{(1)} $$

$$ a^{(1)}=\sigma(z^{(1)}) $$

Hidden layer 2

$$ z^{(2)}=W^{(2)}a^{(1)}+b^{(2)} $$

$$ a^{(2)}=\sigma(z^{(2)}) $$

Output layer

$$ z^{(3)}=W^{(3)}a^{(2)}+b^{(3)} $$

$$ a^{(3)}=\sigma(z^{(3)}) $$

Our prediction is:

$$ \boxed{\hat y=a^{(3)}} $$

The loss function for one training observation is:

$$ L=\frac12\sum\_{j=1}^{n_3}(a_j^{(3)}-y_j)^2 $$

# 3. What do we store during forward propagation?

This is important. We do not want to recalculate everything during backpropagation.

We store the following values:

| Stored value  | Meaning                    | Why needed?                      |
| ------------- | -------------------------- | -------------------------------- |
| $a^{(0)}$ | Input                      | Gradient of first-layer weights  |
| $z^{(1)}$ | Weighted input to hidden 1 | Activation derivative            |
| $a^{(1)}$ | Hidden 1 output            | Gradient of second-layer weights |
| $z^{(2)}$ | Weighted input to hidden 2 | Activation derivative            |
| $a^{(2)}$ | Hidden 2 output            | Gradient of output weights       |
| $z^{(3)}$ | Weighted input to output   | Activation derivative            |
| $a^{(3)}$ | Prediction                 | Loss and output error            |

We also retain the target $y$.

Notice that we store both $z$ and $a$. Technically, if we retain $a$, we can calculate the sigmoid derivative directly from it:

$$ \sigma'(z)=a(1-a) $$

Therefore, storing $z$ is not strictly necessary for sigmoid, but it is useful for other activation functions.

# 4. Backpropagation: work backwards

Now comes the interesting part.

We know the final loss. We want to determine how much each weight and bias contributed to that loss.

We apply the chain rule, starting at the output and moving backwards.

We introduce a useful notation:

$$ \boxed{\delta^{(l)}=\frac{\partial L}{\partial z^{(l)}}} $$

Think of $\delta^{(l)}$ as the error signal arriving at layer $l$, before its activation function.

## Step 1: Output layer

We already know:

$$ L=\frac12\sum_j(a_j^{(3)}-y_j)^2 $$

For each output neuron:

$$ \frac{\partial L}{\partial a^{(3)}} =a^{(3)}-y $$

Using the chain rule:

$$ \delta^{(3)} = \frac{\partial L}{\partial a^{(3)}} \odot \frac{\partial a^{(3)}}{\partial z^{(3)}} $$

Here, $\odot$ denotes element-wise multiplication.

Since sigmoid derivative is:

$$ \sigma'(z)=a(1-a) $$

we get:

$$ \boxed{ \delta^{(3)} =(a^{(3)}-y)\odot a^{(3)}(1-a^{(3)}) } $$

This is the error signal at the output layer.

Now calculate the gradients of weights and biases.

Since:

$$ z^{(3)}=W^{(3)}a^{(2)}+b^{(3)} $$

we obtain:

$$ \boxed{ \nabla\_{W^{(3)}}L =\delta^{(3)}(a^{(2)})^T } $$

$$ \boxed{ \nabla\_{b^{(3)}}L=\delta^{(3)} } $$

Notice that the output layer's weight gradient uses the output of hidden layer 2.

## Step 2: Hidden layer 2

This is where backpropagation becomes interesting.

There is no direct target value for hidden layer 2. So how do we calculate its error?

We propagate the output-layer error backwards through the weights.

The chain rule gives:

$$ \frac{\partial L}{\partial a^{(2)}} = (W^{(3)})^T\delta^{(3)} $$

Why the transpose?

Because we are moving backwards: the weights that connected hidden layer 2 to the output layer now distribute the output error back to the hidden neurons.

But we need the derivative with respect to $z^{(2)}$, not $a^{(2)}$.

Therefore:

$$ \delta^{(2)} = \frac{\partial L}{\partial a^{(2)}} \odot \frac{\partial a^{(2)}}{\partial z^{(2)}} $$

Substituting:

$$ \boxed{ \delta^{(2)} = (W^{(3)})^T\delta^{(3)} \odot a^{(2)}(1-a^{(2)}) } $$

Now calculate the gradients:

$$ \boxed{ \nabla\_{W^{(2)}}L =\delta^{(2)}(a^{(1)})^T } $$

$$ \boxed{ \nabla\_{b^{(2)}}L=\delta^{(2)} } $$

Notice the reuse:

- $W^{(3)}$ is reused to propagate the error backwards.
- $\delta^{(3)}$ is reused from the output layer.
- $a^{(2)}$ is reused to calculate the sigmoid derivative.
- $a^{(1)}$ is reused to calculate the weight gradient.

## Step 3: Hidden layer 1

We repeat exactly the same process.

First propagate the error backwards from hidden layer 2:

$$ \frac{\partial L}{\partial a^{(1)}} = (W^{(2)})^T\delta^{(2)} $$

Apply the sigmoid derivative:

$$ \delta^{(1)} = \frac{\partial L}{\partial a^{(1)}} \odot \frac{\partial a^{(1)}}{\partial z^{(1)}} $$

Therefore:

$$ \boxed{ \delta^{(1)} = (W^{(2)})^T\delta^{(2)} \odot a^{(1)}(1-a^{(1)}) } $$

And the gradients:

$$ \boxed{ \nabla\_{W^{(1)}}L =\delta^{(1)}(a^{(0)})^T } $$

$$ \boxed{ \nabla\_{b^{(1)}}L=\delta^{(1)} } $$

Notice that $a^{(0)}$ is simply the original input $x$.

## Step 4: Input layer (level 1)

There is one important distinction.

The input layer has no trainable weights or biases within this network. Therefore, we do not update the input layer.

However, if we want to calculate how sensitive the loss is to the input, we can do so:

$$ \boxed{ \frac{\partial L}{\partial a^{(0)}} =(W^{(1)})^T\delta^{(1)} } $$

This is useful in applications such as input sensitivity analysis.

# 5. All backpropagation formulas together

This is the complete set of equations for our network.

| Layer      | Error signal ($\delta$)                           | Weight gradient               |
| ---------- | ----------------------------------------------------- | ----------------------------- |
| Output (3) | $(a^{(3)}-y)\odot a^{(3)}(1-a^{(3)})$             | $\delta^{(3)}(a^{(2)})^T$ |
| Hidden 2   | $(W^{(3)})^T\delta^{(3)}\odot a^{(2)}(1-a^{(2)})$ | $\delta^{(2)}(a^{(1)})^T$ |
| Hidden 1   | $(W^{(2)})^T\delta^{(2)}\odot a^{(1)}(1-a^{(1)})$ | $\delta^{(1)}(a^{(0)})^T$ |

For every trainable layer:

$$ \boxed{\nabla\_{b^{(l)}}L=\delta^{(l)}} $$

The input gradient is:

$$ \boxed{\frac{\partial L}{\partial a^{(0)}}=(W^{(1)})^T\delta^{(1)}} $$

# 6. What gets reused? Complete picture

This is perhaps the most important table for understanding the algorithm.

| Calculation                       | Values reused                        |
| --------------------------------- | ------------------------------------ |
| Output delta $\delta^{(3)}$   | $a^{(3)}, y$                     |
| Output weight gradient            | $\delta^{(3)}, a^{(2)}$          |
| Hidden 2 delta $\delta^{(2)}$ | $W^{(3)}, \delta^{(3)}, a^{(2)}$ |
| Hidden 2 weight gradient          | $\delta^{(2)}, a^{(1)}$          |
| Hidden 1 delta $\delta^{(1)}$ | $W^{(2)}, \delta^{(2)}, a^{(1)}$ |
| Hidden 1 weight gradient          | $\delta^{(1)}, a^{(0)}$          |

The key observation:

Forward propagation stores activations. Backpropagation reuses those activations and the weights to calculate error signals and gradients.

# 7. Complete backpropagation algorithm

We can now put everything together.

There are two distinct phases:

- Forward pass: calculate and store activations.
- Backward pass: calculate deltas, gradients and update parameters.

Here is the complete pseudocode for one training observation.

```
# Network:# Input -> Hidden 1 -> Hidden 2 -> Output## sigmoid(z) = 1 / (1 + exp(-z))def sigmoid(z):    return 1 / (1 + exp(-z))def train_one_sample(x, y, W1, b1, W2, b2, W3, b3, eta):    # ==================================    # PHASE 1: FORWARD PROPAGATION    # ==================================    # Input layer    a0 = x    # Hidden layer 1    z1 = W1 @ a0 + b1    a1 = sigmoid(z1)    # Hidden layer 2    z2 = W2 @ a1 + b2    a2 = sigmoid(z2)    # Output layer    z3 = W3 @ a2 + b3    a3 = sigmoid(z3)    # Calculate loss    loss = 0.5 * sum((a3 - y)**2)    # ==================================    # PHASE 2: BACKPROPAGATION
```

Here, `outer(d3, a2)` means the outer product $\delta^{(3)}(a^{(2)})^T$.

The code uses vectorised operations, so it works for multiple neurons in each layer.

# 8. One subtle but important implementation detail

When updating the weights, we must use the original weights to calculate all the backward deltas.

For example:

$$ \delta^{(2)}=(W^{(3)})^T\delta^{(3)}\odot a^{(2)}(1-a^{(2)}) $$

This must use the old $W^{(3)}$, not the already updated $W^{(3)}$.

Similarly, $\delta^{(1)}$ must use the old $W^{(2)}$.

That is why we calculate all gradients first and update all parameters afterwards.

# 9. The entire process at a glance

## Forward pass — left to right

1. Input

   $a^{(0)}=x$
2. Hidden 1

   Calculate and store $z^{(1)},a^{(1)}$
3. Hidden 2

   Calculate and store $z^{(2)},a^{(2)}$
4. Output

   Calculate $z^{(3)},a^{(3)}$, then loss

## Backward pass — right to left

1. Output

   Calculate $\delta^{(3)}$, gradients for $W^{(3)},b^{(3)}$
2. Hidden 2

   Reuse $W^{(3)}$ and $\delta^{(3)}$; calculate $\delta^{(2)}$ and gradients
3. Hidden 1

   Reuse $W^{(2)}$ and $\delta^{(2)}$; calculate $\delta^{(1)}$ and gradients

Update all weights and biases using gradient descent.

The algorithm repeats this process over the training dataset, usually for multiple epochs.

## 10. The most important insight

There is a beautiful recursive structure here.

For any hidden layer $l$:

$$ \boxed{ \delta^{(l)} = (W^{(l+1)})^T\delta^{(l+1)} \odot a^{(l)}(1-a^{(l)}) } $$

And its weight gradient is:

$$ \boxed{ \nabla\_{W^{(l)}}L = \delta^{(l)}(a^{(l-1)})^T } $$

This means we do not need a different derivation for every hidden layer.

We have one general rule that works for any number of layers.

Forward propagation builds the computational history. Backpropagation traverses that history in reverse, reusing stored values to calculate gradients.

That is the essence of backpropagation.

